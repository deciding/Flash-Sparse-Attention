<h1>
  <img src="https://github.com/user-attachments/assets/c1423d38-458b-412b-9e6b-a74379a01eab" alt="logo" width="100" height="100">
  Flash Sparse Attention (FSA)
</h1>

This repository provides the official implementation of **<ins>F</ins>lash <ins>S</ins>parse <ins>A</ins>ttention (FSA)**, which includes a novel kernel design that enables efficient sparse attention computation across a wide range of popular LLMs on modern GPUs.

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Performance](#performance)
* [Citation](#citation)
* [Acknowledgments](#acknowledgments)

## Features

## Installation

## Usage

## Performance

### Kernel Performance

> Performance comparison of Triton-based FSA, NSA, and Full Attention (enabled by Flash Attention) kernels under various configurations. The tuple ($64$, $16$) / ($128$, $8$) represents the block size $BK$ and top-k value $Topk$, respectively.

<img width="4366" height="3057" alt="kernel_perf" src="https://github.com/user-attachments/assets/d1e5868e-ff4c-452f-9810-89495b7ec233" />

### End-to-end Performance

> End-to-end training (right) and prefill (left) latency of FSA, NSA, Full Attention.

<img width="6165" height="3093" alt="e2e_githubpic" src="https://github.com/user-attachments/assets/bb2628b3-2f2a-49fe-8b29-e63027ae043d" />


## Citation

```
@misc{Yan2025FSA,
  title={Flash Sparse Attention: More Efficient Natively Trainable Sparse Attention},
  author={Yan, Ran and Jiang, Youhe and Yuan, Binhang},
  howpublished = {\url{https://github.com/ranyangit/Flash-Sparse-Attention}},
  year={2025}
}
```

## Acknowledgments

[Native Sparse Attention](https://arxiv.org/abs/2502.11089)
