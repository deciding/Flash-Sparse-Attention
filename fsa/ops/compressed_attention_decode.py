import math
import warnings
import torch
import triton

from nsa_ref.ops.compressed_attention import _get_attention_score, transform_score, forward_kernel
from nsa_ref.ops.utils import get_num_warps_stages, is_hopper_gpu
from typing import Any, Tuple, Union

IS_HOPPER_GPU = is_hopper_gpu()


def _compressed_attention_fwd_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor,
    sm_scale: float,
):
    # dtype check
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert k_len == v_len
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # output tensor
    o = torch.zeros_like(q)
    lse = torch.full(
        (num_q_heads, q_len),
        fill_value=-torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    # launch kernel
    grid = lambda META: (
        batch_size,
        num_q_heads,
        triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]),
    )
    BLOCK_SIZE_Q = min(max(16, triton.next_power_of_2(q_len)), 128)
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
    forward_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        lse.stride(0),
        lse.stride(1),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse


def compressed_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    parallel_topk_compute: Union[str, bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention between query and compressed key and value. Compute attention output and topk block idx used in topk_sparse_attention.

    Args:
        q (torch.Tensor): shape [total_q_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        kernel_size (int): kernel size in compress_key_value
        kernel_stride (int): stride of compress_key_value
        block_size (int): key value block size for topk sparse attention.
        topk (int): number of blocks for each query.
        cu_seqlens_q (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        cu_seqlens_k (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_k in flash_attn_func_varlen.
        max_seqlen_q (int): max q len of the batch.
        max_seqlen_k (int): max k len of the batch.
        sm_scale (float, optional): softmax scale. Defaults to None, means 1/sqrt(head_dim).
        init_blocks (int, optional): Number of init blocks for each query. Defaults to 1.
        local_blocks (int, optional): Number of local blocks for each query. Defaults to 2.
        parallel_topk_compute (str, optional): Only set it to False when the sequence length is too long. This can avoid a current bug.
            We'll fix this issue later. Defaults to auto, it will be set to False when the sequence length is greater than 32k and True otherwise.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: attention output and topk_idx used in topk_sparse_attention
    """

    if max_seqlen_q is None:
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    if max_seqlen_k is None:
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()

    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert q.dtype == k.dtype and k.dtype == v.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    # softmax scale
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(q.shape[-1])

    attn_output, lse = _compressed_attention_fwd_decode(
        q,
        k,
        v,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
    )

    # do not select topk index
    if topk <= 0:
        warnings.warn("topk <= 0, returned topk_idx will be None")
        return attn_output, None

    assert topk >= init_blocks + local_blocks
    with torch.no_grad():
        num_k_heads, num_q_heads = k.shape[1], q.shape[1]
        num_shared_q_heads = num_q_heads // num_k_heads
        batch_size = cu_seqlens_q.shape[0] - 1
        q_idx = torch.cat(
            [torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device) for i in range(batch_size)],
            dim=0,
        )
        q_idx = q_idx // block_size

        topk_idx_list = []
        head_tile = 1
        assert num_k_heads % head_tile == 0, f"Num kv heads: {num_k_heads}, head_tile: {head_tile}"
        for h in range(num_k_heads // head_tile):
            # recompute score
            score = _get_attention_score(
                q[:, h * num_shared_q_heads * head_tile : (h + 1) * num_shared_q_heads * head_tile],
                k[:, h * head_tile : (h + 1) * head_tile],
                lse[h * num_shared_q_heads * head_tile : (h + 1) * num_shared_q_heads * head_tile],
                kernel_size,
                kernel_stride,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                sm_scale,
            )
            # transform score to block-wise score
            score = transform_score(
                score,
                kernel_size,
                kernel_stride,
                block_size,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                init_blocks,
                local_blocks,
            )
            # get topk
            topk = min(topk, score.shape[-1])
            if score.dtype == torch.float32:
                score = score.to(torch.bfloat16)
            topk_idx = score.topk(topk, dim=-1, sorted=False).indices
            topk_idx = topk_idx.sort(-1).values

            topk_idx[topk_idx > q_idx[None, :, None]] = -1
            topk_idx = topk_idx.to(torch.int32)
            topk_idx_list.append(topk_idx)
        topk_idx = torch.cat(topk_idx_list, dim=0)

    return attn_output, topk_idx