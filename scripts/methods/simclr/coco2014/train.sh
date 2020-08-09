#! /bin/bash

# Get full path to the config file automatically
full_path=$0
CONFIG_PATH=$(dirname "$full_path")
echo $CONFIG_PATH


python -m selfsup.methods.train --config_path="$CONFIG_PATH"