import time
from contextlib import contextmanager

import torch

E2E_TIME = 0.0


@contextmanager
def cuda_timer(name="block", add=True, meta="", clear=False, mfu_args=None, time_it=True):
    if not time_it:
        yield
        return
    global E2E_TIME
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    end = time.time()

    if mfu_args is not None:
        Ss = mfu_args['seqlens'].tolist()
        num_q_heads = mfu_args['num_q_heads']
        topk = mfu_args['topk']
        C = mfu_args['C']

        causal = 1

        head_dim = mfu_args['head_dim']
        block_size = mfu_args['block_size']
        H = head_dim * num_q_heads
        flops = 0.0
        for S in Ss:
            # Full Attention: 
            #   flops += 4 * S **2 * H / causal
            # FSA / NSA:
            flops += 4 * S * (block_size * topk) * H / causal

        t = end - start

        print(f"[MFU: {name}] ", flops / t / C / 1e12)
        print(f"[MFU args: {name}]", mfu_args)
    if add:
        E2E_TIME += end - start
    print(f"[Timer:{name}] {(end - start) * 1000:.6f} ms; e2e: {E2E_TIME:.6f} s, meta info: {meta}")
    if clear:
        E2E_TIME = 0.0
