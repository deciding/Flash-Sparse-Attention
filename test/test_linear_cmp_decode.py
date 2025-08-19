import argparse
import math
from contextlib import contextmanager
from functools import partial

import torch
import triton

from fsa.ops.FSA_topk_sparse_attention import (_topk_sparse_attention_fwd_opt,
                                               backward_dq_opt)
from fsa.ops.compressed_attention_decode import compressed_attention_decode
from fsa.ops.linear_compress_decode import linear_compress_decode
from nsa_ref.ops import compressed_attention, linear_compress
from nsa_ref.ops.topk_sparse_attention import (_topk_sparse_attention_fwd,
                                               backward_dq)
from nsa_ref.ops.utils import get_num_warps_stages, is_hopper_gpu

IS_HOPPER_GPU = is_hopper_gpu()


def create_cu_seqlens(seqlen: int) -> torch.Tensor:
    """Create cumulative sequence lengths tensor for batch processing."""
    return torch.arange(0, 2 * seqlen, seqlen, dtype=torch.int32)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark topk sparse attention kernels")

    parser.add_argument("--seqlen", type=int, default=65536, help="Sequence length")
    parser.add_argument("--num-q-heads", type=int, default=64, help="Number of query heads")
    parser.add_argument("--num-k-heads", type=int, default=64, help="Number of key/value heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")

    # NSA specific configuration
    parser.add_argument("--block-size", type=int, default=64, help="Block size for sparse attention")
    parser.add_argument("--topk", type=int, default=16, help="Top-k blocks for each query")

    # Benchmark configuration
    parser.add_argument("--warm-up", type=int, default=5, help="Number of warm-up runs")
    parser.add_argument("--benchmark-iters", type=int, default=10, help="Number of benchmark runs")

    return parser.parse_args()


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

    args = parse_args()

    seqlen = args.seqlen
    num_q_heads = args.num_q_heads
    num_k_heads = args.num_k_heads
    head_dim = args.head_dim
    block_size = args.block_size
    topk = args.topk
    kernel_size = 32
    kernel_stride = 16

    # Benchmark parameters
    warm_up = args.warm_up
    benchmark_iters = args.benchmark_iters

    # Create test data
    torch.random.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    assert num_q_heads % num_k_heads == 0, "num_k_heads must be divisible by num_q_heads"
    num_share_q_heads = num_q_heads // num_k_heads

    # Generate random q, k, v tensors
    q = torch.randn(seqlen, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(seqlen, num_k_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen, num_k_heads, head_dim, device=device, dtype=dtype)

    # generate nsa parameters
    compress_key = torch.randn(num_k_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype)
    compress_value = torch.randn(num_k_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype)
    intra_block_pe = torch.randn(num_k_heads, kernel_size, head_dim, device=device, dtype=dtype)

    # Create cumulative sequence lengths
    cu_seqlens = create_cu_seqlens(seqlen).to(device)

    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"cu_seqlens: {cu_seqlens}")
    print(f"kernel_size: {kernel_size}, kernel_stride: {kernel_stride}")

    # Test linear_compress_decode function
    print("\n" + "="*60)
    print("TESTING LINEAR_COMPRESS_DECODE")
    print("="*60)
    
    # Test with different append sizes
    test_append_sizes = [1]
    test_append_sizes = [size for size in test_append_sizes if size <= seqlen//2]
    
    for append_size in test_append_sizes:
        split_point = seqlen - append_size
        if split_point <= kernel_size:  # Need enough tokens for at least one window
            continue
            
        print(f"\nTesting: train({seqlen}) vs train({split_point}) + decode({append_size})")
        
        # Method 1: Full sequence training (ground truth)
        full_compressed_k, full_compressed_cu_seqlen = linear_compress(
            k,
            compress_key,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            intra_block_pe,
        )
        
        full_compressed_v, _ = linear_compress(
            v,
            compress_value,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            None,
        )

        compressed_seqlens_full = full_compressed_cu_seqlen[1:] - full_compressed_cu_seqlen[:-1]
        attn_output_full, topk_idx_full = compressed_attention(
            q=q,
            k=full_compressed_k,
            v=full_compressed_v,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=full_compressed_cu_seqlen,
            max_seqlen_q=seqlen,
            max_seqlen_k=compressed_seqlens_full.max().item(),
            sm_scale=None,
            init_blocks=1,
            local_blocks=2,
            parallel_topk_compute=False,
        )
        
        # Method 2: Split into train(n-k) + decode(k)
        # First part: training on first split_point tokens
        k_first_part = k[:split_point]
        v_first_part = v[:split_point]
        cu_seqlens_first = create_cu_seqlens(split_point).to(device)
        
        compressed_k_first, compressed_cu_seqlens_first = linear_compress(
            k_first_part,
            compress_key,
            cu_seqlens_first,
            kernel_size,
            kernel_stride,
            intra_block_pe,
        )

        compressed_v_first, _ = linear_compress(
            v_first_part,
            compress_value,
            cu_seqlens_first,
            kernel_size,
            kernel_stride,
            None,
        )

        compressed_seqlens_first = compressed_cu_seqlens_first[1:] - compressed_cu_seqlens_first[:-1]
        cu_seqlens_first = cu_seqlens.clone()
        cu_seqlens_first[1] = seqlen - 1
        attn_output_first, topk_idx_first = compressed_attention_decode(
            q=q[:seqlen-1],
            k=compressed_k_first,
            v=compressed_v_first,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            cu_seqlens_q=cu_seqlens_first,
            cu_seqlens_k=compressed_cu_seqlens_first,
            max_seqlen_q=seqlen-1,
            max_seqlen_k=compressed_seqlens_first.max().item(),
            sm_scale=None,
            init_blocks=1,
            local_blocks=2,
            parallel_topk_compute=False,
        )


        
        # Second part: decode on last append_size tokens
        k_last_part = k[split_point:]
        v_last_part = v[split_point:]
        
        # Prepare initial buffer with some context from first part
        # Need last (kernel_size-1) tokens from first part for potential overlapping windows
        buffer_size = min(kernel_size - 1, split_point)
        initial_buffer_k = k_first_part[-buffer_size:] if buffer_size > 0 else None
        initial_buffer_v = v_first_part[-buffer_size:] if buffer_size > 0 else None
        
        # Decode the last part
        decode_k_output, final_len_k, _ = linear_compress_decode(
            k_last_part,
            compress_key,
            kernel_size,
            kernel_stride,
            intra_block_pe,
            split_point,
            initial_buffer_k,
        )
        
        decode_v_output, final_len_v, _ = linear_compress_decode(
            v_last_part,
            compress_value,
            kernel_size,
            kernel_stride,
            None,
            split_point,
            initial_buffer_v,
        )
        
        compressed_cu_seqlens_last = compressed_cu_seqlens_first.clone()
        compressed_cu_seqlens_last[1] += 1
        compressed_seqlens_last = compressed_cu_seqlens_last[1:] - compressed_cu_seqlens_last[:-1]

        cu_seqlens_last = cu_seqlens.clone()
        cu_seqlens_last[1] = 1
        attn_output_last, topk_idx_last = compressed_attention_decode(
            q=q[seqlen-1:],
            k=compressed_k_first,
            v=compressed_v_first,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            cu_seqlens_q=cu_seqlens_last,
            cu_seqlens_k=compressed_cu_seqlens_last,
            max_seqlen_q=1,
            max_seqlen_k=compressed_seqlens_last.max().item(),
            sm_scale=None,
            init_blocks=1,
            local_blocks=2,
            parallel_topk_compute=False,
        )

        # Combine results
        if decode_k_output.shape[0] > 0:
            combined_compressed_k = torch.cat([compressed_k_first, decode_k_output], dim=0)
            combined_compressed_v = torch.cat([compressed_v_first, decode_v_output], dim=0)
            combined_cmp_attn_out = torch.cat([attn_output_first, attn_output_last], dim=0)
        else:
            combined_compressed_k = compressed_k_first
            combined_compressed_v = compressed_v_first       
            combined_cmp_attn_out = attn_output_first
        
        combined_topk_idx = torch.cat([topk_idx_first, topk_idx_last], dim=0)
        
        # Compare results
        print(f"  Full train K shape: {full_compressed_k.shape}, {attn_output_full.shape}, {topk_idx_full.shape}")
        print(f"  Split+decode K shape: {combined_compressed_k.shape}, {combined_cmp_attn_out.shape}, {combined_topk_idx.shape}")
        print(f"  Decode K output shape: {decode_k_output.shape}, {attn_output_first.shape}, {topk_idx_last.shape}")

        
        # Check shapes match
        assert full_compressed_k.shape == combined_compressed_k.shape, \
            f"K shape mismatch: {full_compressed_k.shape} vs {combined_compressed_k.shape}"
        assert full_compressed_v.shape == combined_compressed_v.shape, \
            f"V shape mismatch: {full_compressed_v.shape} vs {combined_compressed_v.shape}"
        
        # Check values match
        torch.testing.assert_close(
            full_compressed_k, combined_compressed_k, 
            rtol=1e-4, atol=1e-5,
            msg="K values don't match between full training and split+decode"
        )
        torch.testing.assert_close(
            full_compressed_v, combined_compressed_v, 
            rtol=1e-4, atol=1e-5,
            msg="V values don't match between full training and split+decode"
        )

        # T
        torch.testing.assert_close(
            attn_output_full, attn_output_first, 
            rtol=1e-4, atol=1e-5,
            msg="Compressed attention values don't match between full training and split+decode"
        )
        
        print((attn_output_full - attn_output_first).abs().max())
        torch.testing.assert_close(
            topk_idx_full, combined_topk_idx, 
            rtol=1e-4, atol=1e-5,
            msg="Topk idx values don't match between full training and split+decode"
        )
        
        print(f"  ✅ Test PASSED for append_size={append_size}")

    
    print("\n" + "="*60)
    print("DECODE TESTING COMPLETED")
    print("="*60)

