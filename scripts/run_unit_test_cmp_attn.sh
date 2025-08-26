# The used GPU device id
device=0
export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=${device} python3 'test/test_cmp_attn_decode.py' \
