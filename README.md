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

<img width="5120" height="3584" alt="e2e_github" src="https://github.com/user-attachments/assets/561c2bbe-b783-4a2e-9cfc-eac2f5d1221b" />

### End-to-end Training Performance

> End-to-end training latency of FSA, NSA, Full Attention.

<img width="3072" height="3072" alt="e2e_training_github" src="https://github.com/user-attachments/assets/2b78d490-95c5-4331-a7e9-36ffec408b57" />

> Loss comparison between FSA, NSA, and Full Attention in end-to-end Llama3-8B model training.

<img width="3072" height="1024" alt="loss_compare" src="https://github.com/user-attachments/assets/7cfbba84-2808-4edd-acc1-f71cdaf75ca2" />


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
