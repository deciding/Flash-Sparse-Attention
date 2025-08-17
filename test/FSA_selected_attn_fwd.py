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

from native_sparse_attention_ref.ops.triton.topk_sparse_attention import (
    _topk_sparse_attention_fwd,
)
from FSA_core.ops.FSA_topk_sparse_attention import (
    _topk_sparse_attention_fwd_opt,
)
from test import cuda_timer

if __name__ == "__main__":
    # [h, n, d]
    inputs = torch.load("topk_sparse_attention_inputs.pt")
    q = inputs['q'].cuda()
    k = inputs['k'].cuda()
    v = inputs['v'].cuda()
    
    topk_idx = inputs['topk_idx'].cuda()
    
    cu_seqlens = inputs['cu_seqlens'].cuda()
    block_size = inputs['block_size']

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

    func = partial(
        _topk_sparse_attention_fwd_opt,
        q,
        k,
        v,
        topk_idx,
        block_size,
        cu_seqlens,
        cu_seqlens,
        N,
        N,
        sm_scale,
        causal=causal,
    )

    for i in range(warm_up):
        print("warm up")
        torch.cuda.reset_peak_memory_stats()
        o, lse, _ = func()
    with cuda_timer("topk_sparse_attention_fwd_opt"):
        for i in range(run):
            torch.cuda.reset_peak_memory_stats()
            print("testing")
            o, lse, _ = func()
            print("Memory allocated opt kernel:", torch.cuda.max_memory_allocated() / 1024**3)

    for i in range(warm_up):
        o_ref, lse_ref = _topk_sparse_attention_fwd(
            q,
            k,
            v,
            topk_idx,
            block_size,
            cu_seqlens,
            cu_seqlens,
            N,
            N,
            sm_scale,
        )
    with cuda_timer("--topk_sparse_attn ref"):
        o_ref, lse_ref = _topk_sparse_attention_fwd(
            q,
            k,
            v,
            topk_idx,
            block_size,
            cu_seqlens,
            cu_seqlens,
            N,
            N,
            sm_scale,
        )

    print("total diff:", (o_ref - o).abs().max())
    print("total realtive diff:", (o_ref - o).abs().max() / o_ref.abs().max())
    print("lse_gap:", (lse_ref - lse).abs().max())


    torch.testing.assert_close(o_ref, o.to(torch.bfloat16), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_ref, lse)
