import math
import torch

from nsa_ref.ops import linear_compress


def linear_compress_decode(
    new_tokens,  # New tokens to append [new_seq_len, num_heads, head_dim]
    compress_weight,  # [num_heads, head_dim * kernel_size, head_dim]
    kernel_size,
    kernel_stride,
    intra_block_pe=None,  # [num_heads, kernel_size, head_dim] or None
    prev_total_len=0,  # Previous total sequence length
    token_buffer=None,  # Buffer containing recent tokens
):
    """
    Decode version that properly handles absolute positions and windowing.
    """
    device = new_tokens.device
    dtype = new_tokens.dtype
    new_seq_len = new_tokens.shape[0]
    
    # Update token buffer with new tokens
    if token_buffer is None:
        all_tokens = new_tokens
        buffer_start_pos = prev_total_len
    else:
        all_tokens = torch.cat([token_buffer, new_tokens], dim=0)
        buffer_start_pos = prev_total_len - token_buffer.shape[0]
    
    new_total_len = prev_total_len + new_seq_len
    
    # Calculate which output positions we had before and what we should have now
    prev_max_output_idx = math.floor((prev_total_len - kernel_size) / kernel_stride) if prev_total_len >= kernel_size else -1
    new_max_output_idx = math.floor((new_total_len - kernel_size) / kernel_stride) if new_total_len >= kernel_size else -1
    
    if new_max_output_idx <= prev_max_output_idx:
        # No new outputs to compute
        # Keep buffer with last (kernel_size + kernel_stride - 1) tokens for future use
        buffer_size = min(kernel_size + kernel_stride - 1, all_tokens.shape[0])
        updated_buffer = all_tokens[-buffer_size:] if all_tokens.shape[0] > 0 else all_tokens
        empty_output = torch.empty(0, new_tokens.shape[1], compress_weight.shape[-1], device=device, dtype=dtype)
        return empty_output, new_total_len, updated_buffer
    
    # We need to compute new outputs
    new_output_indices = list(range(prev_max_output_idx + 1, new_max_output_idx + 1))
    
    # For each new output, determine the input window
    windows_to_compute = []
    for output_idx in new_output_indices:
        window_start_abs = output_idx * kernel_stride  # Absolute position in full sequence
        window_end_abs = window_start_abs + kernel_size
        
        # Convert to relative position in our buffer
        window_start_rel = window_start_abs - buffer_start_pos
        window_end_rel = window_end_abs - buffer_start_pos
        
        # Check if we have all tokens for this window
        if window_start_rel >= 0 and window_end_rel <= all_tokens.shape[0]:
            window_tokens = all_tokens[window_start_rel:window_end_rel]
            windows_to_compute.append(window_tokens)
    
    if not windows_to_compute:
        # We don't have enough tokens in buffer to compute the required windows
        # This shouldn't happen if buffer is managed correctly
        buffer_size = min(kernel_size + kernel_stride - 1, all_tokens.shape[0])
        updated_buffer = all_tokens[-buffer_size:] if all_tokens.shape[0] > 0 else all_tokens
        empty_output = torch.empty(0, new_tokens.shape[1], compress_weight.shape[-1], device=device, dtype=dtype)
        return empty_output, new_total_len, updated_buffer
    
    # Stack windows and compute
    stacked_windows = torch.stack(windows_to_compute, dim=0)  # [num_windows, kernel_size, num_heads, head_dim]
    num_windows = stacked_windows.shape[0]
    
    # Reshape for linear_compress: [total_tokens, num_heads, head_dim]
    compute_tokens = stacked_windows.view(-1, stacked_windows.shape[2], stacked_windows.shape[3])
    
    # Create cu_seqlens for the stacked windows
    cu_seqlens = torch.arange(0, compute_tokens.shape[0] + 1, kernel_size, dtype=torch.int32, device=device)
    
    # Compute compressed representation
    compressed_output, _ = linear_compress(
        compute_tokens,
        compress_weight,
        cu_seqlens,
        kernel_size,
        kernel_size,  # stride = kernel_size for non-overlapping windows
        intra_block_pe,
    )
    
    # Update buffer: keep last (kernel_size + kernel_stride - 1) tokens
    buffer_size = min(kernel_size + kernel_stride - 1, all_tokens.shape[0])
    updated_buffer = all_tokens[-buffer_size:] if all_tokens.shape[0] > 0 else all_tokens
    
    return compressed_output, new_total_len, updated_buffer