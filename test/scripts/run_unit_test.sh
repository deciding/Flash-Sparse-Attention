#!/bin/bash

device=2
export PYTHONPATH=$(pwd)

# Define icons and colors
PASS="✅"
FAIL="❌"
WARN="⚠️"
INFO="ℹ️"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to compare norms
compare_norms() {
    local param_name="$1"
    local norm1="$2"
    local norm2="$3"
    local tolerance=0.001
    
    if [ -z "$norm1" ] || [ -z "$norm2" ]; then
        echo -e "${WARN} ${YELLOW}$param_name: Missing data (norm1=$norm1, norm2=$norm2)${NC}"
        return 1
    fi
    
    # Calculate absolute difference
    diff=$(python3 -c "print(abs($norm1 - $norm2))" 2>/dev/null || echo "999")
    relative_diff=$(python3 -c "print(abs($norm1 - $norm2) / max(abs($norm1), abs($norm2), 1e-10) * 100)" 2>/dev/null || echo "0")
    
    if (( $(echo "$diff < $tolerance" | bc -l) )); then
        echo -e "${PASS} ${GREEN}$param_name: ALIGNED${NC} (diff: $diff, rel: ${relative_diff}%)"
        echo "    NSA: $norm1 | FSA: $norm2"
        return 0
    else
        echo -e "${FAIL} ${RED}$param_name: MISMATCH${NC} (diff: $diff, rel: ${relative_diff}%)"
        echo "    NSA: $norm1 | FSA: $norm2"
        return 1
    fi
}

# Main evaluation loop
for gqa in 16; do
    for seqlen in 65536; do
        for block_size in 64; do
            for topk in 16; do
                # Only run if configuration is valid
                if [[ ($block_size -eq 64 && $topk -eq 16) || ($block_size -eq 128 && $topk -eq 8) ]]; then
                    
                    echo -e "\n${BLUE}🚀 Running Configuration: seqlen=${seqlen}, block-size=${block_size}, topk=${topk}, gqa=${gqa}${NC}"
                    echo "=============================================================================="
                    
                    # Run both use_FSA configurations
                    for use_FSA in false true; do
                        # Build the command
                        cmd="CUDA_VISIBLE_DEVICES=${device} python3 test/test_FSA_module.py \
                            --hidden-size 4096 \
                            --benchmark-iters 5 \
                            --seqlens ${seqlen} \
                            --nseqs 1 \
                            --kv-heads 4 \
                            --gqa-deg ${gqa} \
                            --topk ${topk} \
                            --block-size ${block_size} \
                            --dtype bfloat16 \
                            --kernel-stride 16"
                        
                        # Add conditional flags
                        if [[ "$use_FSA" == "true" ]]; then
                            cmd="$cmd --use-FSA"
                            attn_mode=FSA
                        else
                            attn_mode=NSA
                        fi
                        
                        log_file="unit_test_${attn_mode}_seq${seqlen}_bs${block_size}_topk${topk}_gqa${gqa}.log"
                        
                        if [[ -f "$log_file" ]]; then
                            echo -e "${INFO} Log file already exists: $log_file"
                            echo -e "${WARN} ${YELLOW}Skipping execution for ${attn_mode} (file exists)${NC}"
                        else
                            echo -e "${INFO} Running with ${attn_mode}..."
                            eval $cmd > "$log_file"
                        fi
                        
                        if [ $? -ne 0 ]; then
                            echo -e "${FAIL} ${RED}Test failed for ${attn_mode}${NC}"
                            continue
                        fi
                        
                        echo -e "${PASS} ${GREEN}Test completed for ${attn_mode}${NC}"
                        
                        # Run backward benchmark
                        cmd_bwd="$cmd --benchmark-bwd"
                        log_file_bwd="unit_test_${attn_mode}_seq${seqlen}_bs${block_size}_topk${topk}_gqa${gqa}_bwd.log"
                        
                        if [[ -f "$log_file_bwd" ]]; then
                            echo -e "${INFO} Log file already exists: $log_file"
                            echo -e "${WARN} ${YELLOW}Skipping execution for ${attn_mode} (file exists)${NC}"
                        else
                            echo -e "${INFO} Running backward benchmark with ${attn_mode}..."
                            eval $cmd_bwd > "$log_file_bwd"
                        fi
                        
                        if [ $? -ne 0 ]; then
                            echo -e "${FAIL} ${RED}Backward benchmark failed for ${attn_mode}${NC}"
                        else
                            echo -e "${PASS} ${GREEN}Backward benchmark completed for ${attn_mode}${NC}"
                        fi
                    done
                    
                    # Compare results after both runs are complete
                    NSA_Fwd="unit_test_NSA_seq${seqlen}_bs${block_size}_topk${topk}_gqa${gqa}.log"
                    FSA_Fwd="unit_test_FSA_seq${seqlen}_bs${block_size}_topk${topk}_gqa${gqa}.log"
                    NSA_1F1B="unit_test_NSA_seq${seqlen}_bs${block_size}_topk${topk}_gqa${gqa}_bwd.log"
                    FSA_1F1B="unit_test_FSA_seq${seqlen}_bs${block_size}_topk${topk}_gqa${gqa}_bwd.log"
                    
                    # Check if both forward log files exist
                    if [[ ! -f "$NSA_Fwd" ]] || [[ ! -f "$FSA_Fwd" ]]; then
                        echo -e "${FAIL} ${RED}Missing forward log files for comparison${NC}"
                        echo "NSA_Fwd: $NSA_Fwd (exists: $([ -f "$NSA_Fwd" ] && echo "yes" || echo "no"))"
                        echo "FSA_Fwd: $FSA_Fwd (exists: $([ -f "$FSA_Fwd" ] && echo "yes" || echo "no"))"
                        continue
                    fi
                    
                    # Check if backward benchmark log files exist
                    bwd_logs_exist=true
                    if [[ ! -f "$NSA_1F1B" ]] || [[ ! -f "$FSA_1F1B" ]]; then
                        echo -e "${WARN} ${YELLOW}Missing backward benchmark log files${NC}"
                        echo "NSA_1F1B: $NSA_1F1B (exists: $([ -f "$NSA_1F1B" ] && echo "yes" || echo "no"))"
                        echo "FSA_1F1B: $FSA_1F1B (exists: $([ -f "$FSA_1F1B" ] && echo "yes" || echo "no"))"
                        bwd_logs_exist=false
                    fi
                    
                    echo -e "\n${BLUE}🔍 NSA Norm Comparison Report${NC}"
                    echo "Configuration: seqlen=${seqlen}, block-size=${block_size}, topk=${topk}, gqa=${gqa}"
                    echo "=============================================================================="
                    
                    # Extract and compare Forward output norm
                    echo -e "\n${INFO} Forward Output Comparison:"
                    forward_norm1=$(grep "Forward, output.*norm:" "$NSA_Fwd" | tail -1 | sed 's/.*norm: //')
                    forward_norm2=$(grep "Forward, output.*norm:" "$FSA_Fwd" | tail -1 | sed 's/.*norm: //')
                    compare_norms "Forward Output" "$forward_norm1" "$forward_norm2"
                    forward_aligned=$?
                    
                    # Extract and compare Backward gradient norms
                    echo -e "\n${INFO} Backward Gradient Comparison:"
                    params=("compress_key" "compress_value" "intra_block_pe" "proj_q.weight" "proj_k.weight" "proj_v.weight" "proj_o.weight" "gate.0.weight")
                    
                    aligned_count=0
                    total_count=1  # Start with 1 for forward output
                    if [ $forward_aligned -eq 0 ]; then
                        aligned_count=1
                    fi
                    
                    for param in "${params[@]}"; do
                        norm1=$(grep "Backward, $param.*grad norm:" "$NSA_Fwd" | tail -1 | sed 's/.*grad norm: //')
                        norm2=$(grep "Backward, $param.*grad norm:" "$FSA_Fwd" | tail -1 | sed 's/.*grad norm: //')
                        
                        if [ -n "$norm1" ] && [ -n "$norm2" ]; then
                            total_count=$((total_count + 1))
                            compare_norms "$param" "$norm1" "$norm2"
                            if [ $? -eq 0 ]; then
                                aligned_count=$((aligned_count + 1))
                            fi
                        fi
                    done
                    
                    # Performance comparison
                    echo -e "\n${INFO} Performance Comparison:"
                    
                    # Forward timing
                    NSA_time_fwd=$(grep "\[NSA E2E (Fwd)\] Time:" "$NSA_Fwd" | tail -1 | sed 's/.*Time: //' | sed 's/ ms//')
                    FSA_time_fwd=$(grep "\[FSA E2E (Fwd)\] Time:" "$FSA_Fwd" | tail -1 | sed 's/.*Time: //' | sed 's/ ms//')
                    
                    if [ -n "$NSA_time_fwd" ] && [ -n "$FSA_time_fwd" ]; then
                        time_diff_fwd=$(python3 -c "print(abs($NSA_time_fwd - $FSA_time_fwd))" 2>/dev/null || echo "0")
                        time_ratio_fwd=$(python3 -c "print($FSA_time_fwd / $NSA_time_fwd)" 2>/dev/null || echo "1")

                        if (( $(echo "$time_diff_fwd < 5" | bc -l) )); then
                            echo -e "${PASS} ${GREEN}Forward Performance: SIMILAR${NC}"
                        elif (( $(echo "$FSA_time_fwd < $NSA_time_fwd" | bc -l) )); then
                            echo -e "${PASS} ${GREEN}Forward Performance: FSA FASTER${NC}"
                        else
                            echo -e "${WARN} ${YELLOW}Forward Performance: NSA FASTER${NC}"
                        fi
                    else
                        echo -e "${WARN} ${YELLOW}Forward timing data missing (NSA=${NSA_time_fwd}, FSA=${FSA_time_fwd})${NC}"
                    fi
                    
                    # 1F1B timing and backward calculation (if benchmark logs exist)
                    if [ "$bwd_logs_exist" = true ]; then
                        # Extract 1F1B times (1 forward + 1 backward)
                        NSA_time_1F1B=$(grep "\[NSA E2E (Fwd+Bwd)\] Time:" "$NSA_1F1B" | tail -1 | sed 's/.*Time: //' | sed 's/ ms//')
                        FSA_time_1F1B=$(grep "\[FSA E2E (Fwd+Bwd)\] Time:" "$FSA_1F1B" | tail -1 | sed 's/.*Time: //' | sed 's/ ms//')
                        
                        if [ -n "$NSA_time_1F1B" ] && [ -n "$FSA_time_1F1B" ]; then
                            time_diff_1F1B=$(python3 -c "print(abs($NSA_time_1F1B - $FSA_time_1F1B))" 2>/dev/null || echo "0")
                            time_ratio_1F1B=$(python3 -c "print($FSA_time_1F1B / $NSA_time_1F1B)" 2>/dev/null || echo "1")
                            
                            # Calculate backward-only timing
                            if [ -n "$NSA_time_fwd" ] && [ -n "$FSA_time_fwd" ]; then
                                NSA_time_bwd_only=$(python3 -c "print(max(0, $NSA_time_1F1B - $NSA_time_fwd))" 2>/dev/null || echo "0")
                                FSA_time_bwd_only=$(python3 -c "print(max(0, $FSA_time_1F1B - $FSA_time_fwd))" 2>/dev/null || echo "0")
                                
                                if [ -n "$NSA_time_bwd_only" ] && [ -n "$FSA_time_bwd_only" ] && (( $(echo "$NSA_time_bwd_only > 0" | bc -l) )) && (( $(echo "$FSA_time_bwd_only > 0" | bc -l) )); then
                                    time_diff_bwd=$(python3 -c "print(abs($NSA_time_bwd_only - $FSA_time_bwd_only))" 2>/dev/null || echo "0")
                                    time_ratio_bwd=$(python3 -c "print($FSA_time_bwd_only / $NSA_time_bwd_only)" 2>/dev/null || echo "1")

                                    if (( $(echo "$time_diff_bwd < 5" | bc -l) )); then
                                        echo -e "${PASS} ${GREEN}Backward Performance: SIMILAR${NC}"
                                    elif (( $(echo "$FSA_time_bwd_only < $NSA_time_bwd_only" | bc -l) )); then
                                        echo -e "${PASS} ${GREEN}Backward Performance: FSA FASTER${NC}"
                                    else
                                        echo -e "${WARN} ${YELLOW}Backward Performance: NSA FASTER${NC}"
                                    fi
                                    
                                    if (( $(echo "$time_diff_1F1B < 10" | bc -l) )); then
                                        echo -e "${PASS} ${GREEN}1F1B Performance: SIMILAR${NC}"
                                    elif (( $(echo "$FSA_time_1F1B < $NSA_time_1F1B" | bc -l) )); then
                                        echo -e "${PASS} ${GREEN}1F1B Performance: FSA FASTER${NC}"
                                    else
                                        echo -e "${WARN} ${YELLOW}1F1B Performance: NSA FASTER${NC}"
                                    fi
                                    # Performance breakdown
                                    echo -e "\n${INFO} 📊 Performance Breakdown (ms):"
                                    echo "┌─────────────┬──────────┬──────────┬─────────────┐"
                                    echo "│    Phase    │   NSA    │   FSA    │   Speedup   │"
                                    echo "├─────────────┼──────────┼──────────┼─────────────┤"
                                    printf "│ Forward     │ %7.2f  │ %7.2f  │ %10.2fx │\n" "$NSA_time_fwd" "$FSA_time_fwd" "$(python3 -c "print($NSA_time_fwd / $FSA_time_fwd)" 2>/dev/null || echo "1")"
                                    printf "│ Backward    │ %7.2f  │ %7.2f  │ %10.2fx │\n" "$NSA_time_bwd_only" "$FSA_time_bwd_only" "$(python3 -c "print($NSA_time_bwd_only / $FSA_time_bwd_only)" 2>/dev/null || echo "1")"
                                    printf "│ 1F1B Total  │ %7.2f  │ %7.2f  │ %10.2fx │\n" "$NSA_time_1F1B" "$FSA_time_1F1B" "$(python3 -c "print($NSA_time_1F1B / $FSA_time_1F1B)" 2>/dev/null || echo "1")"
                                    echo "└─────────────┴──────────┴──────────┴─────────────┘"
                                else
                                    echo -e "${WARN} ${YELLOW}Cannot calculate backward-only timing (invalid values)${NC}"
                                    echo -e "${INFO} NSA: 1F1B=${NSA_time_1F1B}ms - Fwd=${NSA_time_fwd}ms = ${NSA_time_bwd_only}ms"
                                    echo -e "${INFO} FSA: 1F1B=${FSA_time_1F1B}ms - Fwd=${FSA_time_fwd}ms = ${FSA_time_bwd_only}ms"
                                fi
                            else
                                echo -e "${WARN} ${YELLOW}Cannot calculate backward-only timing (missing forward data)${NC}"
                            fi
                        else
                            echo -e "${WARN} ${YELLOW}1F1B timing data missing (NSA=${NSA_time_1F1B}, FSA=${FSA_time_1F1B})${NC}"
                        fi
                        
                        # Extract backward memory usage if available
                        NSA_mem_1F1B=$(grep "\[Max allocated\]:" "$NSA_1F1B" | tail -1 | sed 's/.*allocated\]: //')
                        FSA_mem_1F1B=$(grep "\[Max allocated\]:" "$FSA_1F1B" | tail -1 | sed 's/.*allocated\]: //')
                        
                        if [ -n "$NSA_mem_1F1B" ] && [ -n "$FSA_mem_1F1B" ]; then
                            mem_diff_1F1B=$(python3 -c "print(abs($NSA_mem_1F1B - $FSA_mem_1F1B))" 2>/dev/null || echo "0")
                            mem_ratio_bwd=$(python3 -c "print($FSA_mem_1F1B / $NSA_mem_1F1B)" 2>/dev/null || echo "1")
                            mem_percent_diff=$(python3 -c "print(round(($FSA_mem_1F1B - $NSA_mem_1F1B) / $NSA_mem_1F1B * 100, 2))" 2>/dev/null || echo "0")
                        else
                            echo -e "${WARN} ${YELLOW}1F1B memory data missing (NSA=${NSA_mem_1F1B}, FSA=${FSA_mem_1F1B})${NC}"
                        fi
                    else
                        echo -e "${WARN} ${YELLOW}Backward benchmark comparison skipped (missing log files)${NC}"
                    fi
                    
                    # Forward memory comparison (from forward-only runs)
                    echo -e "\n${INFO} 💾 Memory Usage Analysis:"
                    NSA_mem=$(grep "\[Max allocated\]:" "$NSA_Fwd" | tail -1 | sed 's/.*allocated\]: //')
                    FSA_mem=$(grep "\[Max allocated\]:" "$FSA_Fwd" | tail -1 | sed 's/.*allocated\]: //')
                    
                    if [ -n "$NSA_mem" ] && [ -n "$FSA_mem" ]; then
                        mem_diff=$(python3 -c "print(abs($NSA_mem - $FSA_mem))" 2>/dev/null || echo "0")
                        mem_ratio=$(python3 -c "print($FSA_mem / $NSA_mem)" 2>/dev/null || echo "1")
                        mem_percent_diff_fwd=$(python3 -c "print(round(($FSA_mem - $NSA_mem) / $NSA_mem * 100, 2))" 2>/dev/null || echo "0")
                        
                        echo -e "${INFO} Forward Memory Usage: NSA=$(printf "%.2f" $NSA_mem)GB, FSA=$(printf "%.2f" $FSA_mem)GB (FSA uses: $(printf "%.2f" $mem_diff)GB more memory)"
                        echo -e "${INFO} 1F1B Memory Usage: NSA=$(printf "%.2f" $NSA_mem_1F1B)GB, FSA=$(printf "%.2f" $FSA_mem_1F1B)GB (FSA uses: $(printf "%.2f" $mem_diff_1F1B)GB more memory)"
                    else
                        echo -e "${WARN} ${YELLOW}Forward memory data missing (NSA=${NSA_mem}, FSA=${FSA_mem})${NC}"
                    fi
                    
                    # Configuration Summary
                    echo -e "\n${BLUE}📊 Configuration Summary${NC}"
                    echo "=============================================================================="
                    
                    if [ $total_count -gt 0 ]; then
                        alignment_ratio=$(python3 -c "print(round($aligned_count / $total_count * 100, 1))")
                        echo -e "${INFO} Norm Alignment: ${aligned_count}/${total_count} (${alignment_ratio}%)"
                        
                        if [ $aligned_count -eq $total_count ]; then
                            echo -e "${PASS} ${GREEN}CONFIG STATUS: ALL NORMS ALIGNED ✓${NC}"
                        elif [ $aligned_count -gt $((total_count / 2)) ]; then
                            echo -e "${WARN} ${YELLOW}CONFIG STATUS: MOSTLY ALIGNED ⚠️${NC}"
                        else
                            echo -e "${FAIL} ${RED}CONFIG STATUS: MAJOR DIFFERENCES ✗${NC}"
                        fi
                    else
                        echo -e "${FAIL} ${RED}CONFIG STATUS: NO DATA TO COMPARE ✗${NC}"
                    fi
                    
                    # Clean up temporary log files (optional)
                    # rm -f "$NSA_Fwd" "$FSA_Fwd"
                    
                    echo -e "\n${BLUE}Configuration completed.${NC}"
                    echo "=============================================================================="
                    
                fi
            done
        done
    done
done

echo ""
echo -e "${PASS} ${GREEN}🚀 NSA vs FSA Benchmark Analysis Complete!${NC}"
echo "=============================================================================="
