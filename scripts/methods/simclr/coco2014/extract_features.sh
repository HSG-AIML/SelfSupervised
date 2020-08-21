#! /bin/bash

# Get full path to the config file automatically
full_path=$0
CONFIG_PATH=$(dirname "$full_path")
echo $CONFIG_PATH

OUTPUT_PATH="outputs/extracted_features"
TASK_NAME="feature_extraction"
CHECKPOINT_PATH="PATH TO CHECKPOINT"
DATASET="coco2014"
DATASET_PATH="PATH TO DATASET"

# Run the task
python -m selfsup.downstream_tasks.run --task_name "$TASK_NAME" \
                                       --config_path "$CONFIG_PATH" \
                                       --output_path "$OUTPUT_PATH" \
                                       --checkpoint_path "$CHECKPOINT_PATH" \
                                       --dataset "$DATASET" \
                                       --dataset_path "$DATASET_PATH"

