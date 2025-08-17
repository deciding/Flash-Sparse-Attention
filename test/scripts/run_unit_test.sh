device=0

export PYTHONPATH=$(pwd)
for gqa in 1; do
  for seqlen in 32768; do
    for block_size in 64; do
      for topk in 16; do
        for bench_bwd in false; do
          for use_FSA in false true; do
            if [[ ($block_size -eq 64 && $topk -eq 16) || ($block_size -eq 128 && $topk -eq 8) ]]; then
              # Build the command with conditional flags
              cmd="CUDA_VISIBLE_DEVICES=${device} python3 test/test_nsa.py \
                --hidden-size 4096 \
                --seqlens ${seqlen} \
                --nseqs 1 \
                --kv-heads 4 \
                --gqa-deg ${gqa} \
                --topk ${topk} \
                --block-size ${block_size} \
                --dtype bfloat16 \
                --kernel-stride 16"
              
              # Add conditional flags
              if [[ "$bench_bwd" == "true" ]]; then
                cmd="$cmd --bench-bwd"
              fi
              
              if [[ "$use_FSA" == "true" ]]; then
                cmd="$cmd --use-FSA"
              fi
              
              echo "Running with seqlen=${seqlen}, block-size=${block_size}, topk=${topk}, gqa=${gqa}, bench_bwd=${bench_bwd}, use_FSA=${use_FSA}"
              eval $cmd
            fi
          done
        done
      done
    done
  done
done
