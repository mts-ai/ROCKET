#!/bin/bash
# run_sweep.sh

set -e  # Exit on error

BASE_CONFIG="swiftsvd/config/default.yaml"
RESULTS_DIR="./results"
CONFIGS_DIR="./results/configs"
PROFILES_DIR="./results/profiles"

# Define your sweep parameters HERE
MODELS=(
  # "unsloth/Llama-3.2-1B"
  "unsloth/llama-3-8b"
)
RATIOS=(
  # 0.20
  # 0.30
  0.40
  0.50
)

# Create dirs
mkdir -p "$CONFIGS_DIR"

# Loop over all combinations
for model in "${MODELS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    # Sanitize model name for filename
    safe_model=$(echo "$model" | sed 's|/|__|g')
    ratio_str=$(printf "%.2f" "$ratio" | tr '.' '_')
    exp_name="${safe_model}_cr${ratio_str}"
    
    CONFIG_FILE="$CONFIGS_DIR/${exp_name}.yaml"
    
    echo "Generating config for: $exp_name"
    python swiftsvd/generate_configs.py \
      --base_config "$BASE_CONFIG" \
      --model_name "$model" \
      --target_kept_ratio "$ratio" \
      --output_config "$CONFIG_FILE" \
      --results_base_dir "$RESULTS_DIR" \
      --profile_dir "$PROFILES_DIR"
    
    echo "Running experiment: $exp_name"
    swiftsvd-run-pipeline --config "$CONFIG_FILE"
    
    echo "Completed: $exp_name"
    echo "-----------------------------"
  done
done

echo "All experiments finished!"