# This file is modified from the original implementation (implemented by Xunhao Lai)
import argparse
import math
from functools import partial

import torch

from nsa_ref.module import RopeConfig
from nsa_ref.ops import linear_compress
from nsa_ref.ops.flash_attention import flash_attention_varlen
from utils import cuda_timer

if __name__ == "__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=1000)
    parser.add_argument("--seqlens", nargs="+", type=int, default=[1000])
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=-1)
    parser.add_argument("--gqa-deg", type=int, default=1)
    parser.add_argument('--topk', type=int, default=16)
    parser.add_argument('--attn-mode', type=str, default="FSA")
    parser.add_argument("--kernel-size", type=int, default=32)
    parser.add_argument("--kernel-stride", type=int, default=16)
    parser.add_argument("--nseqs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--benchmark-iters", type=int, default=5)
    parser.add_argument("--dtype", type=str,  default="float16", choices=["bfloat16", "float16", "float32"])

    args = parser.parse_args()
    DTYPE = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[args.dtype]
    seqlen = args.seqlens[0]
    head_dim = 128

    if args.kv_heads > 0:
        q_heads = args.kv_heads * args.gqa_deg
        kv_heads = args.kv_heads
    else:
        q_heads = args.heads
        kv_heads = args.heads // args.gqa_deg
    assert q_heads % args.gqa_deg == 0

    from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode

    sparse_attn = (
        FlashSparseAttentionDecode(
            hidden_size=args.hidden_size,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            kernel_size=args.kernel_size,
            kernel_stride=args.kernel_stride,
            block_size=args.block_size,
            topk=args.topk,
            init_blocks=1,
            local_blocks=2,
            window_size=512,
            rope_config=RopeConfig(
                max_position_embeddings=131072,
                head_dim=head_dim,
                rope_theta=500000,
                rope_scaling={
                    "factor": 8.0,
                    "high_freq_factor": 4.0,
                    "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3",
                },
            ),
        )
        .cuda()
        .to(DTYPE)
    )
    print(f"======= Num Heads: {args.attn_mode} =======\n")

    print(f"q_heads={q_heads}, kv_heads={kv_heads}\n")

    print(f"======= Init Moduel: {args.attn_mode} =======\n")
    for name, param in sparse_attn.named_parameters():
        print(f"{args.attn_mode} Parameters, {name}, shape: {param.shape}\n")

    # random input
    if args.nseqs > 1:
        seqlens = torch.LongTensor([seqlen] * args.nseqs).int().cuda()
    else:
        seqlens = torch.LongTensor(args.seqlens).int().cuda()

    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    x = torch.randn(1, args.hidden_size, device="cuda", dtype=DTYPE)

    k_cache = torch.randn(seqlen - 1, args.kv_heads, head_dim, device='cuda', dtype=DTYPE)
    v_cache = torch.randn(seqlen - 1, args.kv_heads, head_dim, device='cuda', dtype=DTYPE)

    cmp_len = (k_cache.shape[0] - args.kernel_size) // args.kernel_stride + 1

    cmp_k_cache = torch.randn(cmp_len, args.kv_heads, head_dim, device='cuda', dtype=DTYPE)
    cmp_k_rope_cache = torch.randn(cmp_len, args.kv_heads, head_dim, device='cuda', dtype=DTYPE)
    cmp_v_cache = torch.randn(cmp_len, args.kv_heads, head_dim, device='cuda', dtype=DTYPE)

    # generate nsa parameters
    compress_key = torch.randn(args.kv_heads, head_dim * args.kernel_size, head_dim, device="cuda", dtype=DTYPE)
    compress_value = torch.randn(args.kv_heads, head_dim * args.kernel_size, head_dim, device="cuda", dtype=DTYPE)
    intra_block_pe = torch.randn(args.kv_heads, args.kernel_size, head_dim, device="cuda", dtype=DTYPE)

    # Compute topk_idx using compressed_attention
    print("Computing topk_idx using compressed_attention...")
    cmp_k_cache, compressed_cu_seqlens = linear_compress(
        k_cache,
        compress_key,
        cu_seqlens,
        args.kernel_size,
        args.kernel_stride,
        intra_block_pe,
    )
    cmp_v_cache, _ = linear_compress(
        v_cache,
        compress_value,
        cu_seqlens,
        args.kernel_size,
        args.kernel_stride,
        None,
    )

    compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    sm_scale = 1 / math.sqrt(head_dim)
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    # warmup
    print(f"======= {args.attn_mode} Forward & Backward Performance Test =======\n")
    for i in range(4):
        y = sparse_attn(x, cu_seqlens_q, cu_seqlens, k_cache, v_cache, cmp_k_cache, cmp_v_cache)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    num_iters = args.benchmark_iters
    for i in range(num_iters):
        y = sparse_attn(x, cu_seqlens_q, cu_seqlens, k_cache, v_cache, cmp_k_cache, cmp_v_cache)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters

    benchmark_mode = "One step decode"
    print(f"[{args.attn_mode} E2E ({benchmark_mode})] Time: {elapsed_ms:.3f} ms\n")

    device = "cuda"
    q = torch.randn(seqlen, q_heads, head_dim, device=device, dtype=DTYPE)
    k = torch.randn(seqlen, kv_heads, head_dim, device=device, dtype=DTYPE)
    v = torch.randn(seqlen, kv_heads, head_dim, device=device, dtype=DTYPE)

    # warmup
    print(f"======= {args.attn_mode} Forward & Backward Performance Test =======\n")
    fa_decode_func = partial(
        flash_attention_varlen,
        q,
        k,
        v,
        torch.tensor([0, 1]).cuda().int(),
        cu_seqlens,
        1,
        seqlens.max().item(),
        False,
    )
    for i in range(4):
        fa_decode_func()

    with cuda_timer("full attention", verbose=False) as fa_timer:
        for i in range(num_iters):
            fa_decode_func()

    fa_time = fa_timer.elapsed_time / num_iters
    print(f"[Full Attention Kernel ({benchmark_mode})] Time: {fa_time:.3f} ms\n")
