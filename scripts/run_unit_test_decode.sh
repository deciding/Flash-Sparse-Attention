# The used GPU device id
device=0
export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=${device} python3 'test/test_FSA_decode.py' \
  --seqlens 65536 \
  --kv-heads 4 \
  --gqa-deg 1 \
  --topk 16 \
  --block-size 64 \
  --kernel-stride 16 \
  --kernel-size 32 \