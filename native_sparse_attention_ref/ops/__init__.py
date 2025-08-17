# Copyright 2025 Xunhao Lai.
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
# See the License for the specific language governing permissions and
# limitations under the License.
from native_sparse_attention_ref.ops.torch.compress_key_value import (
    adaptivepool_compress,
    conv_compress,
    lowrank_compress,
    maxminavgpool_compress,
    maxpool_compress,
    minpool_compress,
    swiglu_compress,
)
from native_sparse_attention_ref.ops.triton.compressed_attention import compressed_attention
from native_sparse_attention_ref.ops.triton.linear_compress import linear_compress
from native_sparse_attention_ref.ops.triton.topk_sparse_attention import (
    topk_sparse_attention,
)
from native_sparse_attention_ref.ops.triton.weighted_pool import (
    avgpool_compress,
    softmaxpool_compress,
    weightedpool_compress,
)

__all__ = [
    "linear_compress",
    "compressed_attention",
    "topk_sparse_attention",
    "avgpool_compress",
    "conv_compress",
    "weightedpool_compress",
    "maxpool_compress",
    "minpool_compress",
    "maxminavgpool_compress",
    "lowrank_compress",
    "swiglu_compress",
    "softmaxpool_compress",
    "adaptivepool_compress",
]
