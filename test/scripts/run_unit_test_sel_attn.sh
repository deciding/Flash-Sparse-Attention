device=3
export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=${device} python3 'test/test_FSA_optimized_sel_attn.py'