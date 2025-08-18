<img width="6200" height="1294" alt="github_title" src="https://github.com/user-attachments/assets/6dccb3a7-735b-4e99-bff9-3c9a31d85649" />

---

This repository provides the official implementation of **<ins>F</ins>lash <ins>S</ins>parse <ins>A</ins>ttention (FSA)**, which includes a novel kernel design that enables efficient sparse attention computation across a wide range of popular LLMs on modern GPUs.

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Performance](#performance)
* [Citation](#citation)
* [Acknowledgments](#acknowledgments)

## Features

## Installation

### Requirements: 
The following requirements should be satisfied:
- [PyTorch](https://pytorch.org/) >= 2.4
- [Triton](https://github.com/openai/triton) >=3.0
- [transformers](https://github.com/huggingface/transformers) >=4.45.0
- [datasets](https://github.com/huggingface/datasets) >=3.3.0
- [accelerate](https://github.com/huggingface/accelerate) >= 1.9.0
- [flash-attn](https://github.com/Dao-AILab/flash-attention) ==2.6.3

You can install dependencies for FSA with:
```sh
pip install -r requirements.txt
```

FSA is currently well tested with: 
1. NVIDIA Ampere or Hopper GPUs (e.g., A100 SXM, H20, H100 SXM, H200 SXM);
2. Datatype of fp16 and bf16;
3. The same head dimension across query, key, and value. 


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
