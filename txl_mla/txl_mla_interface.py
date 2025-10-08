from typing import Tuple
import torch
import triton.language as tl
import txl

from triton.tools.tensor_descriptor import TensorDescriptor

D_Q = 576
D_K = 576
D_V = 512

B_H = 64
B_TOPK = 64    # TopK block size
NUM_THREADS = 128*3
MAX_INIT_VAL = -1e30

# GROUP_SIZE: 8
# NUM_GROUPS = 128/GROUP_SIZE = 16
# NUM_ROWS_PER_GROUP = B_TOPK/NUM_GROUPS = 64/16 = 4
GROUP_SIZE = 8
NUM_GROUPS = 16
NUM_ROWS_PER_GROUP = 4

@txl.jit
def sparse_prefill_kernel(
    # dimensions
    s_q: tl.constexpr,
    s_kv: tl.constexpr,
    h_q: tl.constexpr,
    h_kv: tl.constexpr,
    d_qk: tl.constexpr,
    d_v: tl.constexpr,
    topk: tl.constexpr,

    # scaling factors
    sm_scale: tl.constexpr,
    sm_scale_log2: tl.constexpr,

    # input tensors
    q_desc,                       # *bf16
    kv_desc,                      # *bf16
    indices_ptr,                 # *i32

    # strides
    q_stride0: tl.constexpr, q_stride1: tl.constexpr,
    kv_stride0: tl.constexpr, kv_stride1: tl.constexpr,
    ind_stride0: tl.constexpr, ind_stride1: tl.constexpr,

    # outputs
    out_desc,                            # *bf16
    max_logits_desc,                     # *fp32
    lse_desc,                            # *fp32,
):
    pid = tl.program_id(axis=0);
    q_h_idx = pid % (h_q//B_H)
    s_q_idx = pid // (h_q//B_H)

    wgid = txl.warpgroup_id()
    wid = txl.warp_id()
    tid_wg = txl.tid(0) % 128

    sQ_buf = txl.smem_alloc([B_H, D_Q], dtype=tl.bfloat16, num_stages=1); sQ = txl.get_buffer(sQ_buf, 0)
    sO_buf = txl.smem_alloc([B_H, D_V], dtype=tl.bfloat16, num_stages=1); sO = txl.get_buffer(sO_buf, 0)
    sS_buf = txl.smem_alloc([B_H, B_TOPK], dtype=tl.bfloat16, num_stages=1); sS0 = txl.get_buffer(sS_buf, 0); sS1 = txl.get_buffer(sS_buf, 1)
    sK_buf = txl.smem_alloc([B_TOPK, D_K], dtype=tl.bf16, num_stages=2); sK0 = txl.get_buffer(sK_buf, 0); sK1 = txl.get_buffer(sK_buf, 1)
    sV0l = txl.smem_slice(sK0, 0, D_V//2, dim=1); sV0r = txl.smem_slice(sK0, D_V//2, D_V//2, dim=1);
    sV1l = txl.smem_slice(sK1, 0, D_V//2, dim=1); sV1r = txl.smem_slice(sK1, D_V//2, D_V//2, dim=1);

    # split: WG0, WG1
    is_kv_valid_buf = txl.smem_alloc([B_TOPK], dtype=tl.bool, num_stages=2); is_kv_valid_wg0 = txl.get_buffer(is_kv_valid_buf, 0); is_kv_valid_wg1 = txl.get_buffer(is_kv_valid_buf, 1)

    bar_q_buf = txl.mbar_alloc(1, num_stages=1); bar_q = txl.get_buffer(bar_q_buf, 0)
    bar_k0_free_buf = txl.mbar_alloc(128, num_stages=2); bar_k0_free_l = txl.get_buffer(bar_k0_free_buf, 0); bar_k0_free_r = txl.get_buffer(bar_k0_free_buf, 1);
    bar_k0_ready_buf = txl.mbar_alloc(128, num_stages=2); bar_k0_ready_l = txl.get_buffer(bar_k0_ready_buf, 0); bar_k0_ready_r = txl.get_buffer(bar_k0_ready_buf, 1);

    bar_k1_free_buf = txl.mbar_alloc(128, num_stages=2); bar_k1_free_l = txl.get_buffer(bar_k1_free_buf, 0); bar_k1_free_r = txl.get_buffer(bar_k1_free_buf, 1);
    bar_k1_ready_buf = txl.mbar_alloc(128, num_stages=2); bar_k1_ready_l = txl.get_buffer(bar_k1_ready_buf, 0); bar_k1_ready_r = txl.get_buffer(bar_k1_ready_buf, 1);


    # TODO: why 16?
    bar_is_kv_valid_ready_buf = txl.mbar_alloc(16, num_stages=1); bar_is_kv_valid_ready = txl.get_buffer(bar_is_kv_valid_ready_buf, 0)
    # TODO: make this auto
    layout_is_kv_valid: tl.constexpr = txl.DistributedLinearLayout(
        reg_bases = [[1], [0], [8], [16], [32]],
        lane_bases = [[2], [4], [0], [0], [0]],
        warp_bases = [[0], [0]],
        block_bases = [],
        shape = [64, ]
    )
    sM_buf = txl.smem_alloc([64], dtype=tl.float32, num_stages=1); sM = txl.get_buffer(sM_buf, 0)
    #layout_sM: tl.constexpr = txl.BlockedLayout(size_per_thread=[2], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    # NOTE: difference of sM and sL, is that whether 2 wg are involved, please change warp_bases accordingly
    layout_sM: tl.constexpr = txl.DistributedLinearLayout(
        reg_bases = [[1]],
        lane_bases = [[0], [0], [2], [4], [8]],
        warp_bases = [[16], [32]],
        block_bases = [],
        shape = [64, ]
    )
    sL_buf = txl.smem_alloc([128], dtype=tl.float32, num_stages=1); sL = txl.get_buffer(sL_buf, 0)
    layout_sL: tl.constexpr = txl.DistributedLinearLayout(
        reg_bases = [[1]],
        lane_bases = [[0], [0], [2], [4], [8]],
        warp_bases = [[16], [32]],
        block_bases = [],
        shape = [64, ]
    )

    final_max_logits_buf = txl.smem_alloc([64], dtype=tl.float32, num_stages=1); final_max_logits = txl.get_buffer(final_max_logits_buf, 0)
    final_lse_buf = txl.smem_alloc([64], dtype=tl.float32, num_stages=1); final_lse = txl.get_buffer(final_lse_buf, 0)

    num_topk_blocks = topk // B_TOPK

    if txl.is_warpgroup([0,1]):
        txl.reg_alloc(216)

        txl.mbar_expect(bar_q, B_H * D_Q * 2)
        txl.tma_load(bQ, q_desc, [s_q_idx, q_h_idx, 0], bar_q)

        rM = MAX_INIT_VAL
        rL = 0.0 # TODO: 2 row
        rO = 0.0 # TODO: 2 row

        txl.mbar_wait(bar_q, 0) # phase 0
        mbar_wait_phase = 0


        for block_idx in range(0, num_topk_blocks, 2):
            if block_idx == 0:
                ## pipelined_wait_and_qkt_gemm_l
                ## pipelined_wait_and_qkt_gemm_r
                ## wait0
                # TODO: left and right
                # TODO: split
                # TODO: group commit
                if txl.is_warpgroup([0]):
                    txl.mbar_wait(bar_k0_ready_l, mbar_wait_phase)
                    txl.mbar_wait(bar_k0_ready_r, mbar_wait_phase)
                    rP = tl.dot(sQ, sK0) # [B_H, D_K]
                if txl.is_warpgroup([1]):
                    # NOTE: also find r first then l
                    txl.mbar_wait(bar_k1_ready_l, mbar_wait_phase)
                    txl.mbar_wait(bar_k1_ready_r, mbar_wait_phase)
                    rP = tl.dot(sQ, sK1) # [B_H, D_K]
                txl.dot_wait(0)

            ## mask rP

            txl.mbar_wait(bar_is_kv_valid_ready, mbar_wait_phase)
            # TODO: rP vs. is_kv_valid
            if txl.is_warpgroup([0]):
                reg_is_kv_valid_wg0 = txl.smem_load(is_kv_valid_wg0, layout_is_kv_valid)
                rP = tl.where(reg_is_kv_valid_wg0, rP, float('-inf'))
            if txl.is_warpgroup([1]):
                reg_is_kv_valid_wg1 = txl.smem_load(is_kv_valid_wg1, layout_is_kv_valid)
                rP = tl.where(reg_is_kv_valid_wg1, rP, float('-inf'))


            if txl.is_warpgroup([1]):
                txl.bar_wait(8, 256) # wg0_bunch_0_ready

            ## online softmax and rescale o

            bar_is_kv_valid_ready.wait(mbar_wait_phase)
            scale = sm_scale_log2

            if txl.is_warpgroup([1]):
                r_sM = txl.frag_smem_load(sM, layout_sM)

            cur_max = tl.max(rP, axis=-1) # TODO reg max, only reduce 4
            cur_max *= scale

            # For WG1, old_max comes from sM (written by WG0); for WG0, old_max comes from rM (read by WG0 from sM in the last round)
            if txl.is_warpgroup([0]):
                new_maxs = tl.maximum(rM, cur_max)
            if txl.is_warpgroup([1]):
                new_maxs = tl.maximum(r_sM, cur_max)
            scale_for_o = tl.exp2(rM - new_maxs)
            rO *= scale_for_o

            rP = tl.exp2(rP * scale - new_maxs)
            rS = rP.to(tl.bfloat16)

            cur_sum = tl.sum(rP, axis=-1)
            rL = rL * scale_for_o + cur_sum

            txl.frag_smem_store(sM, new_maxs, layout_sM)
            rM = new_maxs


            if txl.is_warpgroup([0]):
                txl.bar_arrive(8, 256) # wg0_bunch_0_ready
                rO = tl.dot(rS, sV0l, rO) # only use 1 wg

                txl.dot_wait(0)

                # mark V0l as free
                bar_k0_free_l.arrive()

                txl.bar_wait(9, 256) # wg1_bunch_0_ready

                new_rM = txl.frag_smem_load(sM, layout_sM) # 2 per thread
                scale_factors = txl.exp2(rM - new_rM)
                rM = new_rM


                ## scale_rS

                rS *= scale_factors

            if txl.is_warpgroup([1]):
                txl.bar_arrive(9, 256) # wg1_bunch_0_ready
                rO = tl.dot(rS, sV1r, rO)


            ## save_rS_to_sS(rS, sS0/1, idx in wg)

            if txl.is_warpgroup([0]):
                txl.smem_store(sS0, rS)
                txl.bar_arrive(10, 256) # wg0_s0_ready
                txl.bar_wait(11, 256) # wg1_s1_ready
            if txl.is_warpgroup([1]):
                txl.smem_store(sS1, rS)
                txl.bar_wait(10, 256) # wg0_s0_ready


            if txl.is_warpgroup([0]):
                ## rescale_rO
                rO *= scale_factors
                rL *= scale_factors # TODO: not same encoding

            if txl.is_warpgroup([0]):
                rO = tl.dot(sS1, sV1l, rO)
            if txl.is_warpgroup([1]):
                rO = tl.dot(sS0, sV0r, rO)
                txl.bar_arrive(11, 256)

            mbar_wait_phase ^= 1

            if txl.is_warpgroup([0]):
                if (block_idx + 2 < num_topk_blocks):
                    ## pipelined_wait_and_qkt_gemm_l
                    txl.mbar_wait(bar_k0_ready_l, mbar_wait_phase)
                    txl.mbar_wait(bar_k0_ready_r, mbar_wait_phase)
                    rP = tl.dot(sQ, sK0) # [B_H, D_K]

                    txl.dot_wait(0)
                    txl.mbar_arrive(bar_k1_free_l)
                else: # last iter
                    txl.dot_wait(0)
                    txl.mbar_arrive(bar_k1_free_l)

            if txl.is_warpgroup([1]):
                    txl.dot_wait(1)
                    txl.mbar_arrive(bar_k1_free_r)


        ## outer block_idx
        ## reduceL
        reduced_rL = tl.sum(rL) # TODO: assume reduce 4 threads
        txl.frag_smem_store(sL, rL, layout_sL)
        txl.bar_wait(15, 256) # sL_ready, TODO: needed?
        peer_L = txl.frag_smem_load(sL, layout_sL) # TODO: which wg, change layout_sL accordingly
        reduced_rL += peer_L

        ## store_O
        scale_factors = tl.where(rL == 0.0, 1.0, 1.0/rL)
        sO_l = txl.smem_slice(sO, 0, D_V//2, dim=1)
        rO = rO * scale_factors
        # B_H * 64 is one tile, 128 threads
        # rO each tile: (2, (2, 2, 4)), 32 values
        # each tile is further split into 4 splits, each with 16 values, each split to 2x8
        # sO is to make each wg into 64 (B_H) slots, e.g. 0/16 in the same slot
        # each thread has 4 stsm_addrs, each with one split of 8 values
        # step1, 128 threads work together to save rO into sO, 8 values 4 times
        txl.smem_store(sO_l, rO) # TODO: by tile_idx with size 64?
        # step2: use tma to store sO to O, tile by tile
        out_desc.store([s_q_idx, q_h_idx, 0], sO_l) # 0 is wg0

        # TODO: desc store with smem as input
        # WG1 do the lse store
        if txl.is_warpgroup([1]):
            # TODO: row_idx = (idx_in_warpgroup/32)*16 + local_row_idx*8 + (idx_in_warpgroup%32/4);
            is_no_valid_tokens = rL == 0.0
            reg_logits = tl.where(is_no_valid_tokens, float('-inf'), rM)
            reg_lse = tl.where(is_no_valid_tokens, float('-inf'), tl.log2(rL) + rM)
            txl.frag_smem_store(final_max_logits, reg_logits, layout_sM)
            txl.frag_smem_store(final_lse, reg_lse, layout_sM)
            max_logits_desc.store([s_q_idx, q_h_idx], final_max_logits)
            lse_desc.store([s_q_idx, q_h_idx], final_lse)


    if txl.is_warpgroup([2]):
        txl.reg_dealloc(72)
        # GROUP_SIZE: 8
        # NUM_GROUPS = 128/GROUP_SIZE = 16
        # NUM_ROWS_PER_GROUP = B_TOPK/NUM_GROUPS = 64/16 = 4

        idx_in_group = idx_in_warpgroup % 8
        group_idx = idx_in_warpgroup // 8

        gIndices = indices_ptr + s_q_idx * topk

        # gIndices = indices + sq_idx * topk
        # local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row
        # offs = (topk_block_idx+buf_idx)*B_TOPK + local_row*NUM_GROUPS + topk_group_idx;
        # i.e. which topk block + which block row + which row, suppose, 64 rows are 16 by 16, each use 8 threads
        # NOTE: assert that h_kv == 1
        # let's do this: given a block_idx, calculate offsets0, then add B_TOPK to make it offsets1
        # offsets0:
        # base: indices_ptr + s_q_idx * topk
        # for tid, block_idx
        # 0 16 32 48 x8, 1 17 33 49 x8, ... x16
        threads = tl.arange(0, 128)
        groups = threads // GROUP_SIZE
        rows = tl.arange(0, B_TOPK, NUM_GROUPS)
        offs = groups[:, None] + rows[None, :]
        offs = tl.reshape(offs, (128 * NUM_ROWS_PER_GROUP,))

        # my_sKV_base: k0, 64x64, (gid, idx_in_group*8) : starting from the first, each thread working on 8 cols
        # my_gKV_base = kv, idx_in_group * 8
        # given block_idx, buf_idx, tile_start, tile_end
        # tile index is from 0 to 8, (0 -> 576)
        # my_sKV_base + (buf_idx * B_TOPK*D_K + tile_idx*(B_TOPK*64) + local_row * NUM_GROUPS * 64 (within tile)
        # my_gKV_base + token_indices[buf_idx][local_row] + tile_idx * 64

        my_sKV_base = txl.smem_slice(sK0, group_idx, 64, dim=1)
        my_gKV_base = kv_ptr + idx_in_group * 8
        #TODO: how async_load arrange the cp.async.cg
        #TODO: load tile by tile
        for block_idx in range(0, num_topk_blocks, 2):
            ## load_token_indices
            token_indices0 = tl.load(gIndices + block_idx * B_TOPK, offs)
            is_token_valid0 = token_indices0 >= 0 and token_indices0 < s_kv
            token_indices0 *= kv_stride0

            token_indices1 = tl.load(gIndices + (block_idx+1) * B_TOPK, offs)
            is_token_valid1 = token_indices1 >= 0 and token_indices1 < s_kv
            token_indices1 *= kv_stride0

            for local_row in range(0, NUM_ROWS_PER_GROUP):
                token_index = token_indices0
                txl.smem_slice(sK0, )
                txl.async_load()

            pass


    # 1. confirm layout 2. make sure the complicated layout is logically needed

def sparse_prefill_fwd(q, kv, indices, sm_scale, d_v):
    s_q = q.size(0)
    s_kv = kv.size(0)
    h_q = q.size(1)
    h_kv = kv.size(1)
    d_qk = q.size(2) # kv.size(2) is the same
    topk = indices.size(2)

    assert h_kv == 1
    assert topk % (2*B_TOPK) == 0
    assert topk > 0
    assert h_q % B_H == 0

    q_stride0, q_stride1 = q.stride(0), q.stride(1)
    kv_stride0, kv_stride1 = kv.stride(0), kv.stride(1)
    ind_stride0, ind_stride1 = indices.stride(0), indices.stride(1)

    sm_scale_log2 = sm_scale * 1.44269504

    out = torch.empty((s_q, h_q, d_v), dtype=q.dtype, device=q.device)
    max_logits = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)
    lse = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)

    dummy_block3d = [1, 1, 1]
    dummy_block2d = [1, 1]

    q_desc = TensorDescriptor(q, q.shape, q.stride(), dummy_block3d)
    kv_desc = TensorDescriptor(kv, kv.shape, kv.stride(), dummy_block3d)
    indices_desc = TensorDescriptor(indices, indices.shape, indices.stride(), dummy_block3d)
    out_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block3d)
    max_logits_desc = TensorDescriptor(max_logits, max_logits.shape, max_logits.stride(), dummy_block2d)
    lse_desc = TensorDescriptor(lse, lse.shape, lse.stride(), dummy_block2d)

    grid = [h_q//B_H * s_q, 1, 1]
    num_warpgroups = 3

    sparse_prefill_kernel[grid](
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale_log2,
        q_desc, kv_desc, indices,
        q_stride0, q_stride1,
        kv_stride0, kv_stride1,
        ind_stride0, ind_stride1,
        out_desc, max_logits_desc, lse_desc, num_warpgroups=num_warpgroups
    )

    return out, max_logits, lse

def txl_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512

    Returns:
        (output, max_logits, lse)
        About the definition of output, max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
    """
    results = sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v
    )
    return results

