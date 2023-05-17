#!/bin/bash

# Retrieve the directory of the script
SCRIPT_DIR="$(dirname "$0")"
DATA_DIR="$SCRIPT_DIR/../data"

# Activate virtual environment
source "$SCRIPT_DIR/../.venv/Scripts/activate"

# Check if 'do_pretrain' is set to true
if [ "$1" = "pretrain" ]; then
    echo "Generating data for pretraining"
    sh "$DATA_DIR/pretraining/preprocess_data.sh"
elif [ "$1" = "eval" ]; then
    echo "Generating data for evaluation"
    sh "$DATA_DIR/evaluation/preprocess_data.sh"
elif [ -z "$1" ]; then
    echo "Generating data for pretraining and evaluation"
    sh "$DATA_DIR/pretraining/preprocess_data.sh"
    sh "$DATA_DIR/evaluation/preprocess_data.sh"
else
    echo "Invalid argument"
    exit 1
fi

# Deactivate virtual environment
deactivate
