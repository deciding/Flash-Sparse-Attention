import itertools
# This file is for benchmarking full attention (enabled by flash-attn-triton)
from functools import partial

import torch
import triton

try:
    # NOTE(yiakwy) : for general scenario, we should use FA3 as baseline
    import flash_attn_interface as fa3
    flash_attention_varlen_fa3 = fa3.flash_attn_varlen_func
except:
    raise Exception("We only use FA3 in Hopper platform as formal baseline.")

# NOTE (yiakwy) : CUDNN (v1.14) is not the SOTA in attention decoding
import cudnn

from nsa_ref.ops.flash_attention import \
    flash_attention_varlen as flash_attention_varlen_naive_triton


def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(-1, -1)):
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (-1, -1):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device='cuda')
            col_left = torch.maximum(row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0))
            col_right = torch.minimum(row_idx + seqlen_k - seqlen_q - window_size[1], torch.tensor(seqlen_k - 1))
            avg_seqlen = (col_right - col_left + 1).float().mean().item()

    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)


def get_bench_input_configs(head_dim=128):
    gqa = [1, 2]
    seq_lens = [32768, 66536]
    seq_lens = [16 * 1024, 32 * 1024, 64 * 1024]

    num_heads = [40,]
    hidden_size = [ head_dim * H for H in num_heads ]

    # Note(yiakwy) : Tri Dao's implementation only supports bf16
    dtype = [ torch.bfloat16, ]

    benchmark_bwd = [ True, False ]

    configs = list( itertools.product(gqa, seq_lens, hidden_size, num_heads, dtype, benchmark_bwd) )
    return configs


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["gqa", "seq_lens", "hidden_size", "num_heads", "dtype", "benchmark_bwd"],
        x_vals=get_bench_input_configs(),
        line_arg="provider",
        line_vals=["fa3", "naive_triton", "fa3_perf"],
        line_names=[
            "FA3 (ms)",
            "Naive Triton (ms)",
            "FA3 (TFLOPS)"
        ],
        styles=[("blue", "-"), ("green", "-"), ("blue", "None")],
        ylabel="miu s",  # "elapse",
        plot_name="bf16-fp8-full-attention-performance",
        args={},
    )
)
def benchmark_flashattn_varlen(
    gqa,
    seq_lens,
    hidden_size,
    num_heads,
    dtype,
    benchmark_bwd,
    provider,
    model=None,
    args=None,
    head_dim=128
):
    num_kv_heads = num_heads // gqa
    if type(seq_lens) is not list:
        seq_lens = [seq_lens,]
    print(f"testing GQA={gqa}, benchmark_bwd={benchmark_bwd}, Ss={seq_lens}, heads={num_heads} provider={provider} ...", flush=True)

    if provider == "fa3" or provider == "fa3_perf":
        flash_attention_varlen = flash_attention_varlen_fa3
    else:
        flash_attention_varlen = flash_attention_varlen_naive_triton

    assert hidden_size % num_heads == 0
    head_dim = hidden_size // num_heads
    device = 'cuda'

    batch_size = len(seq_lens)
    total_tokens = sum(seq_lens)

    def init_(total_tokens, num_heads, head_dim, dtype, device, requires_grad):
        indata = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
        if dtype == torch.float8_e4m3fn:
            indata = indata.to(torch.float8_e4m3fn)
        return indata

    # 创建 packed q, k, v（[total_tokens, nheads, headdim]）
    q = init_(total_tokens, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = init_(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = init_(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device, requires_grad=True)

    # 创建 cu_seqlens: 前缀和索引
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0)

    max_seqlen = max(seq_lens)

    func = partial(
        flash_attention_varlen,
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    # warmup
    for _ in range(10):
        o = func()
        if benchmark_bwd:
            vo_grad = torch.randn_like(q, dtype=torch.bfloat16)
            if q.dtype == torch.float8_e4m3fn:
                vo_grad = vo_grad.to(torch.float8_e4m3fn)
            torch.autograd.backward(o, vo_grad)

    torch.cuda.synchronize()

    if benchmark_bwd:
        o = func()
        vo_grad = torch.randn_like(q, dtype=torch.bfloat16)
        if q.dtype == torch.float8_e4m3fn:
            vo_grad = vo_grad.to(torch.float8_e4m3fn)
        fn = lambda: o.backward(vo_grad, retain_graph=True)
    else:
        fn = func

    quantiles = [0.5, 0.2, 0.8]

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=10, rep=10)

    if provider == "fa3_perf":
        # by default
        nflops = 2 * batch_size * num_heads * seq_lens[0] * ( seq_lens[0] / 2 ) * (head_dim + head_dim)
        tflops = nflops / (ms * 1e-3) * 1e-12
        min_tflops = nflops / (min_ms * 1e-3) * 1e-12
        max_tflops = nflops / (max_ms * 1e-3) * 1e-12
        return tflops, min_tflops, max_tflops
    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark_flashattn_varlen.run(print_data=True)
