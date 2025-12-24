#!/bin/bash
# run_sweep.sh

set -e  # Exit on error

BASE_CONFIG="swiftsvd/config/base_config.yaml"
RESULTS_DIR="./results"
CONFIGS_DIR="./results/configs"

# Define your sweep parameters HERE
MODELS=(
  "unsloth/Llama-3.2-1B"
  "meta-llama/Llama-3.1-8B"
)
RATIOS=(
  0.10
  0.20
  0.30
)

# Create dirs
mkdir -p "$CONFIGS_DIR"

# Loop over all combinations
for model in "${MODELS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    # Sanitize model name for filename
    safe_model=$(echo "$model" | sed 's|/|__|g')
    ratio_str=$(printf "%.2f" "$ratio" | tr '.' '_')
    exp_name="${safe_model}_r${ratio_str}"
    
    CONFIG_FILE="$CONFIGS_DIR/${exp_name}.yaml"
    
    echo "Generating config for: $exp_name"
    python swiftsvd/generate_configs.py \
      --base_config "$BASE_CONFIG" \
      --model_name "$model" \
      --target_kept_ratio "$ratio" \
      --output_config "$CONFIG_FILE" \
      --results_base_dir "$RESULTS_DIR"
    
    echo "Running experiment: $exp_name"
    swiftsvd-run-pipeline --config "$CONFIG_FILE"
    
    echo "Completed: $exp_name"
    echo "-----------------------------"
  done
done

echo "All experiments finished!"