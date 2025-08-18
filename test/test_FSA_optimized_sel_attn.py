import math
import torch
import time

from contextlib import contextmanager
from functools import partial
from FSA_core.ops.FSA_topk_sparse_attention import (
    _topk_sparse_attention_fwd_opt,
)
from native_sparse_attention_ref.ops.triton.topk_sparse_attention import (
    _topk_sparse_attention_fwd,
)
from native_sparse_attention_ref.ops import compressed_attention, linear_compress

def create_cu_seqlens(batch_size: int, seqlen: int) -> torch.Tensor:
    """Create cumulative sequence lengths tensor for batch processing."""
    return torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32)


@contextmanager
def cuda_timer(name):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    class TimeCapture:
        def __init__(self):
            self.elapsed_time = 0
    
    time_capture = TimeCapture()
    
    try:
        yield time_capture
    finally:
        end_event.record()
        torch.cuda.synchronize()
        time_capture.elapsed_time = start_event.elapsed_time(end_event)
        print(f"[{name}] Time: {time_capture.elapsed_time:.3f} ms")


if __name__ == "__main__":
    # Test configuration
    batch_size = 1
    seqlen = 65536
    num_heads = 64
    head_dim = 128
    block_size = 64
    topk = 16
    kernel_size = 32
    kernel_stride = 16
    
    # Create test data
    torch.random.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    
    total_len = batch_size * seqlen
    
    # Generate random q, k, v tensors
    q = torch.randn(total_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_len, num_heads, head_dim, device=device, dtype=dtype)

    # generate nsa parameters
    compress_key = torch.randn(num_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype)
    compress_value = torch.randn(num_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype)
    intra_block_pe = torch.randn(num_heads, kernel_size, head_dim, device=device, dtype=dtype)

    # Create cumulative sequence lengths
    cu_seqlens = create_cu_seqlens(batch_size, seqlen).to(device)
    
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"cu_seqlens: {cu_seqlens}")
    print(f"block_size: {block_size}, topk: {topk}")
    
    # Compute topk_idx using compressed_attention
    print("Computing topk_idx using compressed_attention...")

    # compressed key and value before rope
    compressed_k, compressed_cu_seqlens = linear_compress(
        k,
        compress_key,
        cu_seqlens,
        kernel_size,
        kernel_stride,
        intra_block_pe,
    )
    compressed_v, _ = linear_compress(
        v,
        compress_value,
        cu_seqlens,
        kernel_size,
        kernel_stride,
        None,
    )

    # attention between query and compressed key value
    compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    sm_scale = 1 / math.sqrt(head_dim)
    
    # Get attention output and topk_idx from compressed attention
    _, topk_idx = compressed_attention(
        q=q,
        k=compressed_k,
        v=compressed_v,
        kernel_size=kernel_size,
        kernel_stride=kernel_stride,
        block_size=block_size,
        topk=topk,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=compressed_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=compressed_seqlens.max().item(),
        sm_scale=None,
        init_blocks=1,
        local_blocks=2,
        parallel_topk_compute=False,
    )
    
    H, N, TopK = topk_idx.shape
    num_blocks = topk_idx.max().item() + 1
    causal = (topk_idx == -1).sum().item() != 0
    
    print(f"topk_idx shape: {topk_idx.shape}, dtype: {topk_idx.dtype}")
    print(f"topk_idx range: [{topk_idx.min().item()}, {topk_idx.max().item()}]")
    print(f"H, N, TopK: {H}, {N}, {TopK}")
    print(f"num_blocks: {num_blocks}")
    print(f"causal: {causal}")

    # Benchmark parameters
    warm_up = 5
    run = 10
    
    # Create partial function for optimized version
    func_opt = partial(
        _topk_sparse_attention_fwd_opt,
        q,
        k,
        v,
        topk_idx,
        block_size,
        cu_seqlens,
        cu_seqlens,
        seqlen,
        seqlen,
        sm_scale,
        causal=causal,
    )

    # Warm up optimized version
    print("\nWarming up FSA optimized kernel...")
    for i in range(warm_up):
        torch.cuda.reset_peak_memory_stats()
        o_opt, lse_opt, _ = func_opt()
    
    # Benchmark optimized version
    print("Benchmarking FSA optimized kernel...")
    torch.cuda.reset_peak_memory_stats()
    with cuda_timer("topk_sparse_attention_fwd_opt") as timer:
        for i in range(run):
            o_opt, lse_opt, _ = func_opt()
    
    opt_time = timer.elapsed_time
    opt_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Memory allocated opt kernel: {opt_memory:.3f} GB")

    # Warm up reference version
    print("\nWarming up reference kernel...")
    for i in range(warm_up):
        o_ref, lse_ref = _topk_sparse_attention_fwd(
            q,
            k,
            v,
            topk_idx,
            block_size,
            cu_seqlens,
            cu_seqlens,
            seqlen,
            seqlen,
            sm_scale,
        )

    # Benchmark reference version
    print("Benchmarking reference kernel...")
    torch.cuda.reset_peak_memory_stats()
    with cuda_timer("topk_sparse_attention_fwd_ref") as timer_ref:
        for i in range(run):
            o_ref, lse_ref = _topk_sparse_attention_fwd(
                q,
                k,
                v,
                topk_idx,
                block_size,
                cu_seqlens,
                cu_seqlens,
                seqlen,
                seqlen,
                sm_scale,
            )

    ref_time = timer_ref.elapsed_time
    ref_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Memory allocated ref kernel: {ref_memory:.3f} GB")

    # Accuracy comparison
    print("\n" + "="*50)
    print("ACCURACY COMPARISON")
    print("="*50)
    
    total_diff = (o_ref - o_opt).abs().max()
    relative_diff = total_diff / o_ref.abs().max()
    lse_diff = (lse_ref - lse_opt).abs().max()
    
    print(f"Output absolute diff: {total_diff:.6e}")
    print(f"Output relative diff: {relative_diff:.6e}")
    print(f"LSE absolute diff: {lse_diff:.6e}")

    # Accuracy comparison
    print("\n" + "="*50)
    print("Performance COMPARISON")
    print("="*50)

    avg_opt_time = opt_time / run
    avg_ref_time = ref_time / run
    print(f"Reference kernel latency: {avg_ref_time:.3f} ms")
    print(f"Optimized kernel latency: {avg_opt_time:.3f} ms")
    print(f"Latency ratio (ref/ours): {(avg_ref_time / avg_opt_time):.3f}x")

    # Memory comparison
    print("\n" + "="*50)
    print("MEMORY COMPARISON")
    print("="*50)
    print(f"Reference kernel memory: {ref_memory:.3f} GB")
    print(f"Optimized kernel memory: {opt_memory:.3f} GB")
    print(f"Memory difference: {(opt_memory - ref_memory):.3f} GB")
    print(f"Memory ratio (opt/ref): {(opt_memory / ref_memory):.3f}x")

    # Assert accuracy
    try:
        torch.testing.assert_close(o_ref, o_opt.to(torch.bfloat16), atol=1e-2, rtol=1e-2)
        print(f"\n✅ Output accuracy test PASSED (atol=1e-2, rtol=1e-2)")
    except AssertionError as e:
        print(f"\n❌ Output accuracy test FAILED: {e}")
    
    try:
        torch.testing.assert_close(lse_ref, lse_opt, atol=1e-5, rtol=1e-5)
        print(f"✅ LSE accuracy test PASSED (atol=1e-5, rtol=1e-5)")
    except AssertionError as e:
        print(f"❌ LSE accuracy test FAILED: {e}")

    # Summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seqlen}")
    print(f"  - Num heads: {num_heads}")
    print(f"  - Head dim: {head_dim}")
    print(f"  - Block size: {block_size}")
    print(f"  - TopK: {topk}")
    print(f"  - Kernel size: {kernel_size}")
    print(f"  - Kernel stride: {kernel_stride}")
    print(f"  - Causal: {causal}")
    print(f"\nAccuracy:")
    print(f"  - Output diff: {total_diff:.2e} (relative: {relative_diff:.2e})")
    print(f"  - LSE diff: {lse_diff:.2e}")
    print(f"\nPerformance:")
    print(f"  - Reference: {avg_ref_time:.3f} ms")
    print(f"  - Optimized: {avg_opt_time:.3f} ms")
    print(f"  - Ratio (ref/opt): {(avg_ref_time / avg_opt_time):.3f}x")
    print(f"\nMemory:")
    print(f"  - Reference: {ref_memory:.3f} GB")
    print(f"  - Optimized: {opt_memory:.3f} GB")
    print(f"  - Ratio (opt/ref): {(opt_memory / ref_memory):.3f}x")
    
    print("\n🎉 Benchmark completed successfully!")
