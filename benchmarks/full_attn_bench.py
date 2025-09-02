import itertools

# This file is for benchmarking full attention (enabled by flash-attn-triton)
from functools import partial

import math

import torch
import triton

try:
    # NOTE(yiakwy) : for general scenario, we should use FA3 as baseline
    import flash_attn_interface as fa3
    flash_attention_varlen_fa3 = fa3.flash_attn_varlen_func
except:
    raise Exception("We only use FA3 in Hopper platform as formal baseline.")
    
from nsa_ref.ops.flash_attention import flash_attention_varlen as flash_attention_varlen_naive_triton

# NOTE (yiakwy) : CUDNN (v1.14) is not the SOTA in attention decoding
import cudnn

# adopted from flash attnetion v3
def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")

# CUDNN computing graph executor setup, adapted from flash attnetion v3
def cudnn_spda_setup(q, k, v, causal=False, window_size_left=-1):
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    assert v.shape == (b, nheads_k, seqlen_k, headdim)
    assert cudnn is not None, 'CUDNN is not available'
    q_gpu, k_gpu, v_gpu = q, k, v
    o_gpu = torch.empty_like(q_gpu)
    stats_gpu = torch.empty(b, nheads, seqlen_q, 1, dtype=torch.float32, device=q.device)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT, # we can switch softmax intermediate to bf16 in inferences
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())

    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        is_inference=False,
        attn_scale=1.0 / math.sqrt(headdim),
        # use_causal_mask_bottom_right=causal or window_size_left >= 0,
        use_causal_mask=causal or window_size_left >= 0,
        sliding_window_length=window_size_left if window_size_left >= 0 and not causal else None,
    )

    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        stats: stats_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return o_gpu

    return run


def cudnn_spda_bwd_setup(q, k, v, o, g, lse, causal=False, window_size_left=-1):
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    assert v.shape == (b, nheads_k, seqlen_k, headdim)
    assert g.shape == (b, nheads, seqlen_q, headdim)
    assert o.shape == (b, nheads, seqlen_q, headdim)
    assert lse.shape == (b, nheads, seqlen_q, 1)
    assert cudnn is not None, 'CUDNN is not available'
    q_gpu, k_gpu, v_gpu, o_gpu, g_gpu = q, k, v, o, g
    dq_gpu = torch.empty_like(q_gpu)
    dk_gpu = torch.empty_like(k_gpu)
    dv_gpu = torch.empty_like(v_gpu)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())
    o = graph.tensor_like(o_gpu.detach())
    g = graph.tensor_like(g_gpu.detach())
    stats = graph.tensor_like(lse.detach())

    dq, dk, dv = graph.sdpa_backward(
        name="sdpa_backward",
        q=q,
        k=k,
        v=v,
        o=o,
        dO=g,
        stats=stats,
        attn_scale=1.0 / math.sqrt(headdim),
        # use_causal_mask_bottom_right=causal or window_size_left >= 0,
        use_causal_mask=causal or window_size_left >= 0,
        sliding_window_length=window_size_left if window_size_left >= 0 and not causal else None,
    )

    dq.set_output(True).set_dim(dq_gpu.shape).set_stride(dq_gpu.stride())
    dk.set_output(True).set_dim(dk_gpu.shape).set_stride(dk_gpu.stride())
    dv.set_output(True).set_dim(dv_gpu.shape).set_stride(dv_gpu.stride())

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        g: g_gpu,
        stats: lse,
        dq: dq_gpu,
        dk: dk_gpu,
        dv: dv_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return dq_gpu, dk_gpu, dv_gpu

    return run


def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(-1, -1)):
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (-1, -1):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device='cuda')
            col_left = torch.maximum(row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0))
            col_right = torch.minimum(row_idx + seqlen_k - seqlen_q - window_size[1], torch.tensor(seqlen_k - 1))
            avg_seqlen = (col_right - col_left + 1).float().mean().item()

    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)


def get_bench_input_configs(head_dim=128):
    gqa = [1, 2]
    seq_lens = [32768, 66536]
    seq_lens = [16 * 1024, 32 * 1024, 64 * 1024]

    num_heads = [40,]
    hidden_size = [ head_dim * H for H in num_heads ]

    # Note(yiakwy) : Tri Dao's implementation only supports bf16
    dtype = [ torch.bfloat16, ]

    benchmark_bwd = [ False, True ]

    configs = list( itertools.product(gqa, seq_lens, hidden_size, num_heads, dtype, benchmark_bwd) )
    return configs


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["gqa", "seq_lens", "hidden_size", "num_heads", "dtype", "benchmark_bwd"],
        x_vals=get_bench_input_configs(),
        line_arg="provider",
        line_vals=["fa3", "naive_triton", "cudnn", "fa3_perf"],
        line_names=[
            "FA3 (ms)",
            "Naive Triton (ms)",
            "CUDNN (ms)",
            "FA3 (TFLOPS)"
        ],
        styles=[("blue", "-"), ("green", "-"), ("black", "-"), ("blue", "None")],
        ylabel="miu s",  # "elapse",
        plot_name="bf16-fp8-full-attention-performance",
        args={},
    )
)
def benchmark_flashattn_varlen(
    gqa,
    seq_lens,
    hidden_size,
    num_heads,
    dtype,
    benchmark_bwd,
    provider, 
    model=None, 
    args=None, 
    head_dim=128
):
    num_kv_heads = num_heads // gqa
    if type(seq_lens) is not list:
         # Note (yiakwy) : this is essentially batch size 1 test case exclusively for inference scenario
        seq_lens = [seq_lens,]
    print(f"testing GQA={gqa}, benchmark_bwd={benchmark_bwd}, Ss={seq_lens}, heads={num_heads} provider={provider} ...", flush=True)

    if provider == "fa3" or provider == "fa3_perf":
        flash_attention_varlen = flash_attention_varlen_fa3
    else:
        flash_attention_varlen = flash_attention_varlen_naive_triton

    assert hidden_size % num_heads == 0
    head_dim = hidden_size // num_heads
    device = 'cuda'

    batch_size = len(seq_lens)
    total_tokens = sum(seq_lens)

    def init_(total_tokens, num_heads, head_dim, dtype, device, requires_grad):
        indata = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
        if dtype == torch.float8_e4m3fn:
            indata = indata.to(torch.float8_e4m3fn)
        return indata

    # creating packed q, k, v of shape=(total_tokens, nheads, headdim）
    q = init_(total_tokens, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = init_(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = init_(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device, requires_grad=True)

    # creating cu_seqlens indcies
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0)

    max_seqlen = max(seq_lens)

    if provider == "cudnn":
        assert all(s == seq_lens[0] for s in seq_lens)
        max_seqlen = seq_lens[0]
        q = q.view(batch_size, max_seqlen, num_heads, head_dim) # requires seq_lens has equal
        k = k.view(batch_size, max_seqlen, num_kv_heads, head_dim)
        v = v.view(batch_size, max_seqlen, num_kv_heads, head_dim)

        if benchmark_bwd:
            o = torch.randn(batch_size, max_seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            lse = torch.randn(batch_size, max_seqlen, num_heads, 1, device=device, dtype=torch.float32)
        else:
            func = cudnn_spda_setup(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True, window_size_left=-1)
    else:
        func = partial(
            flash_attention_varlen,
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
        if provider == "cudnn":
            vo_grad = torch.randn_like(q, dtype=torch.bfloat16)
            if q.dtype == torch.float8_e4m3fn:
                vo_grad = vo_grad.to(torch.float8_e4m3fn)
            fn = cudnn_spda_bwd_setup(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), o.transpose(1, 2), vo_grad.transpose(1, 2), lse.transpose(1, 2), causal=True, window_size_left=-1)
        else:
            o = func()
            vo_grad = torch.randn_like(q, dtype=torch.bfloat16)
            if q.dtype == torch.float8_e4m3fn:
                vo_grad = vo_grad.to(torch.float8_e4m3fn)
            fn = lambda: o.backward(vo_grad, retain_graph=True)
    else:
        fn = func

    # warmup
    for _ in range(10):
        fn()

    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=10, rep=10)

    if provider == "fa3_perf":
        # by default
        nflops = 2 * batch_size * num_heads * seq_lens[0] * ( seq_lens[0] / 2 ) * (head_dim + head_dim)
        tflops = nflops / (ms * 1e-3) * 1e-12
        min_tflops = nflops / (min_ms * 1e-3) * 1e-12
        max_tflops = nflops / (max_ms * 1e-3) * 1e-12
        return tflops, min_tflops, max_tflops
    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark_flashattn_varlen.run(print_data=True)