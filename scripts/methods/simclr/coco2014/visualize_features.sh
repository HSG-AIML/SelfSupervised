#! /bin/bash

# Get full path to the config file automatically
full_path=$0
CONFIG_PATH=$(dirname "$full_path")
echo $CONFIG_PATH


TASK_NAME="feature_visualization"
CONFIG_PATH=""

OUTPUT_PATH="outputs/feature_visualization"
FEATURES_PATH="outputs/extracted_features/features.npy"
LABELS_PATH="outputs/extracted_features/labels.npy"
DIM_REDUCT_METHOD="umap"

# Run the task
python -m selfsup.downstream_tasks.run --task_name "$TASK_NAME" \
                                       --config_path "$CONFIG_PATH" \
                                       --output_path "$OUTPUT_PATH" \
                                       --features_path "$FEATURES_PATH" \
                                       --labels_path "$LABELS_PATH" \
                                       --dim_reduct_method "$DIM_REDUCT_METHOD"