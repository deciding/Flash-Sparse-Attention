# Copyright 2025 Ran Yan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific
import math
from functools import partial

import torch
import triton

from native_sparse_attention_ref.ops.triton.topk_sparse_attention import backward_dq
from FSA_core.ops.FSA_topk_sparse_attention import backward_dq_opt
from native_sparse_attention_ref.ops.triton.utils import get_num_warps_stages, is_hopper_gpu
from test import cuda_timer

IS_HOPPER_GPU = is_hopper_gpu()


if __name__ == "__main__":
    # [h, n, d]
    inputs = torch.load("topk_sparse_attention_input_bwd.pt")

    # Load inputs
    q = inputs['q'].cuda()
    k = inputs['k'].cuda()
    v = inputs['v'].cuda()
    topk_idx = inputs['topk_idx'].cuda()
    lse = inputs['lse'].cuda()
    delta = inputs['delta'].cuda()
    do = inputs['do'].cuda()
    cu_seqlens_q = inputs['cu_seqlens_q'].cuda()
    cu_seqlens_k = inputs['cu_seqlens_k'].cuda()

    num_k_heads = inputs['num_k_heads']
    block_size = inputs["block_size"]
    num_share_q_heads = inputs['num_share_q_heads']
    head_dim = inputs['head_dim']
    topk = inputs['topk']
    permute_results = inputs['permute_results']

    for i in range(len(permute_results)):
        for key, val in permute_results[i].items():
            if isinstance(val, list):
                permute_results[i][key] = [t.cuda() for t in val]
            elif isinstance(val, torch.Tensor):
                permute_results[i][key] = val.cuda()

    H, N, TopK = topk_idx.shape
    D = 128
    num_blocks = topk_idx.max().item() + 1
    causal = (topk_idx == -1).sum().item() != 0
    
    print("topk_idx", topk_idx.shape, topk_idx.dtype, topk_idx.min(), topk_idx.max())
    print("H, N, Topk", H, N, TopK)
    print("num_blocks", num_blocks)
    print("causal:", causal)

    warm_up = 5
    run = 1

    torch.random.manual_seed(42)
    sm_scale = 1 / math.sqrt(D)

    dq = torch.zeros_like(q)

    func = partial(
        backward_dq_opt,
        q,
        k,
        v,
        topk_idx,
        lse,
        delta,
        do,
        dq,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        topk,
        sm_scale,
        block_size,
        permute_results,
    )

    for i in range(warm_up):
        print("warm up")
        torch.cuda.reset_peak_memory_stats()
        dq = func()
    with cuda_timer("topk_sparse_attention_bwd_opt"):
        for i in range(run):
            torch.cuda.reset_peak_memory_stats()
            print("testing")
            dq = func()
            print("Memory allocated opt kernel:", torch.cuda.max_memory_allocated() / 1024**3)

    # compute dq
    dq_ref = torch.zeros_like(q)
    num_q_loop = N // 32768 + 1  # calculate multiple querys in one kernel if seqlence length is too long
    grid = (1, num_k_heads, triton.cdiv(N, num_q_loop))
    BLOCK_SIZE_K = block_size
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_H = max(16, triton.next_power_of_2(num_share_q_heads))
    BLOCK_SIZE_T = triton.next_power_of_2(topk)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K, IS_HOPPER_GPU)

    for i in range(num_warps):
        backward_dq[grid](
            q,
            k,
            v,
            topk_idx,
            lse,
            delta,
            do,
            dq_ref,
            cu_seqlens_q,
            cu_seqlens_k,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            topk,
            num_q_loop,
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
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            lse.stride(0),
            lse.stride(1),
            delta.stride(0),
            delta.stride(1),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            dq_ref.stride(0),
            dq_ref.stride(1),
            dq_ref.stride(2),
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_T=BLOCK_SIZE_T,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    with cuda_timer("--bwd dq"):
        backward_dq[grid](
            q,
            k,
            v,
            topk_idx,
            lse,
            delta,
            do,
            dq_ref,
            cu_seqlens_q,
            cu_seqlens_k,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            topk,
            num_q_loop,
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
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            lse.stride(0),
            lse.stride(1),
            delta.stride(0),
            delta.stride(1),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            dq_ref.stride(0),
            dq_ref.stride(1),
            dq_ref.stride(2),
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_T=BLOCK_SIZE_T,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    print("total diff:", (dq_ref - dq).abs().max())
    print("total realtive diff:", (dq_ref - dq).abs().max() / dq_ref.abs().max())
    torch.testing.assert_close(dq_ref, dq.to(torch.bfloat16), atol=6e-4, rtol=1e-3)
