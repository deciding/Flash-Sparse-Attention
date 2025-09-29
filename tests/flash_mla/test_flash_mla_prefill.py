import math
import time
from typing import Tuple
import random
import dataclasses

import torch
import triton

from flash_mla import flash_mla_sparse_fwd
from tests.flash_mla.lib import check_is_allclose

@dataclasses.dataclass
class TestParam:
    b: int
    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 576
    d_v: int = 512
    seed: int = 0
    check_correctness: bool = True
    benchmark: bool = True

@dataclasses.dataclass
class Testcase:
    t: TestParam
    q: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor

def generate_testcase(t: TestParam) -> Testcase:
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    random.seed(t.seed)
    q = torch.randn((t.b, t.s_q, t.h_q, t.d_qk), dtype=torch.bfloat16)/10
    kv = torch.randn((t.b, t.s_kv, t.h_kv, t.d_qk), dtype=torch.bfloat16)/10

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((t.b, t.s_q, t.h_kv, t.topk), t.s_kv, dtype=torch.int32)
    for b in range(t.b):
        for s in range(t.s_q):
            for h in range(t.h_kv):
                # NOTE We use the following method to generate indices so that most indices lies within [s_kv-20000, s_kv), which is more realistic for sparse attention
                near_mask = torch.randint(0, 32, (min(t.topk, t.s_kv),)) < 31
                cur_indices = torch.randperm(t.s_kv)[:t.topk]
                cur_indices[near_mask] = torch.randint(max(0, t.s_kv-20000), t.s_kv-1, (near_mask.sum().item(),))
                if len(cur_indices) < t.topk:
                    cur_indices = torch.cat([cur_indices, torch.full((t.topk - len(cur_indices),), 2147480000)])
                cur_indices = cur_indices[torch.randperm(t.topk)]
                indices[b, s, h] = cur_indices
    indices = indices.to(q.device)

    return Testcase(
        t=t,
        q=q,
        kv=kv,
        indices=indices
    )

def get_flop(p: TestParam) -> float:
    flop = 2 * sum([
        p.h_q * p.d_qk * p.topk,
        p.h_q * p.d_v * p.topk
    ]) * p.b * p.s_q
    return flop

def reference_torch(p: TestParam, t: Testcase, sm_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)
    
    assert p.b == 1
    indices = t.indices[0, :, 0, :] # [s_q, topk]
    invalid_indices_mask = (indices < 0) | (indices >= p.s_kv)
    qs = t.q[0, :, :, :].float()  # [s_q, h_q, d_qk]
    kvs = t.kv[0, :, 0, :].float()  # [s_kv, d_qk]

    kvs = torch.index_select(kvs, 0, indices.masked_fill(invalid_indices_mask, 0).flatten()).view(p.s_q, p.topk, p.d_qk)  # [s_q, topk, d_qk]
    attn_score = qs @ kvs.transpose(1, 2)    # [s_q, h_q, topk]
    attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float('-inf'))
    attn_score *= sm_scale * math.log2(math.e)
    max_logits = torch.max(attn_score, dim=-1)[0]   # [s_q, h_q]
    lse = log2sumexp2(attn_score, dim=-1)   # [s_q, h_q]
    attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))   # [s_q, h_q, topk]
    result = attn_score @ kvs[:, :, :p.d_v]
    return (max_logits, lse, result)

@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()
    assert p.b == 1

    t = generate_testcase(p)
    sm_scale = 1 / math.sqrt(p.d_qk)
    torch.cuda.synchronize()

    def run_ans():
        return flash_mla_sparse_fwd(
            t.q.squeeze(0), t.kv.squeeze(0), t.indices.squeeze(0), sm_scale=sm_scale
        )
    
    ans_out, ans_max_logits, ans_lse = run_ans()
    torch.cuda.synchronize()

    if p.benchmark:
        flop = get_flop(p)
        prefill_ans_time: float = triton.testing.do_bench(run_ans, warmup=10, rep=20)/1000  # type: ignore
        prefill_flops = flop/prefill_ans_time/1e12
        print(f"Prefill:  {prefill_ans_time*1e6:4.0f} us, {prefill_flops:.3f} TFlops")

    if p.check_correctness:
        torch.cuda.synchronize()
        ref_max_logits, ref_lse, ref_out = reference_torch(p, t, sm_scale)
        torch.cuda.synchronize()

        is_correct = True
        is_correct &= check_is_allclose("out", ans_out, ref_out, abs_tol=8e-4, rel_tol=2.01/128, cos_diff_tol=7e-6)
        is_correct &= check_is_allclose("max_logits", ans_max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01/65536)
        is_correct &= check_is_allclose("lse", ans_lse, ref_lse, abs_tol=1e-6, rel_tol=2.01/65536)

        return is_correct
    else:
        return True


def main():
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    correctness_cases = [
        # Regular shapes
        TestParam(1, s_q, s_kv, topk, h_q=128, benchmark=False)
        for s_kv, topk in [
            # Regular shapes
            (128, 128),
            (256, 256),
            (512, 512),

            # Irregular shapes
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            # Irregular shapes with OOB TopK
            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [
            1, 62
        ]
    ]

    corner_cases = [
        # In these cases, some blocks may not have any valid topk indices
        TestParam(1, s_q, s_kv, topk, h_q=128, benchmark=False)
        for s_kv, topk in [
            (32, 2048),
            (64, 8192)
        ]
        for s_q in [1, 1024]
    ]

    performance_cases = [
        TestParam(1, s_q, s_kv, topk, h_q=128)
        for s_q in [4096]
        for s_kv in [4096, 8192, 16384, 32768, 49152, 65536, 81920, 98304, 114688, 131072]
        for topk in [2048]
    ]

    testcases = correctness_cases + corner_cases + performance_cases

    failed_cases = []
    for test in testcases:
        if test.benchmark:
            time.sleep(0.2)
        is_correct = run_test(test)
        if not is_correct:
            failed_cases.append(test)
    
    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
    else:
        print(f"\033[32m\033[1mAll {len(testcases)} cases passed!\033[0m")

if __name__ == '__main__':
    main()
