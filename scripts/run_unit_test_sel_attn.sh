# The used GPU device id
device=0
export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=${device} python3 'test/test_FSA_optimized_sel_attn.py' \
    --seqlen 65536 \
    --num-q-heads 64 \
    --num-k-heads 64 \
    --head-dim 128 \
    --block-size 64 \
    --topk 16
