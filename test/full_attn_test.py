# This file is for benchmarking full attention (enabled by flash-attn-triton)
import torch
from native_sparse_attention_ref.ops.triton.flash_attention import flash_attention_varlen


def benchmark_flashattn_varlen(seq_lens=[16384] * 1, hidden_size=4096, num_heads=64, num_kv_heads=4, dtype=torch.bfloat16, benchmark_bwd=False):
    assert hidden_size % num_heads == 0
    head_dim = hidden_size // num_heads
    device = 'cuda'

    batch_size = len(seq_lens)
    total_tokens = sum(seq_lens)

    # 创建 packed q, k, v（[total_tokens, nheads, headdim]）
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    vo_grad = torch.randn_like(q)

    # 创建 cu_seqlens: 前缀和索引
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0)

    max_seqlen = max(seq_lens)

    # warmup
    for _ in range(10):
        o = flash_attention_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )
        if benchmark_bwd:
            torch.autograd.backward(o, vo_grad)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(10):
        o = flash_attention_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )
        if benchmark_bwd:
            torch.autograd.backward(o, vo_grad)

    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / 10

    print(f"[varlen] Total tokens: {total_tokens}, Max seq: {max_seqlen}, Time: {elapsed_ms:.3f} ms")
    return elapsed_ms / 1000  # seconds


if __name__ == "__main__":
    for gqa in [1, 2]:
        for benchmark_bwd in [True]:
            for n in [40]:
                for Ss in [32768, 66536]:
                    Ss = [Ss]
                    head_dim = 128
                    n_kv = n // gqa
                    H = head_dim * n
                    # for causal
                    K = 0.5

                    print(f"GQA={gqa}, benchmark_bwd={benchmark_bwd}, Ss={Ss}, heads={n}", flush=True)
                    
                    t = benchmark_flashattn_varlen(seq_lens=Ss, hidden_size=H, num_heads=n, num_kv_heads=n_kv, benchmark_bwd=benchmark_bwd, dtype=torch.bfloat16)

                    flops = 0.0
                    for S in Ss:
                        flops += 4 * S**2 * H / 1e12 * K if not benchmark_bwd else 4 * S**2 * H / 1e12 * K * 3
                    print(f"[varlen] FLOPs: {flops:.3f} TFLOPs")
                    MFU = flops / t / 312
                    print(f"[varlen] MFU: {MFU:.3f}")
