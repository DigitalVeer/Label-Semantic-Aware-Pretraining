#!/bin/bash

# Retrieve the directory of the script
SCRIPT_DIR="$(dirname "$0")"

# Activate virtual environment
source "$SCRIPT_DIR/../../.venv/Scripts/activate"

python "$SCRIPT_DIR/polyai-bank/get_data.py"
python "$SCRIPT_DIR/wikihow/get_data.py"

# Delete preprocessed data
rm -rf "$SCRIPT_DIR/preprocessed_data"
rm -rf "$SCRIPT_DIR/dataset/csv"
rm -rf "$SCRIPT_DIR/dataset/json"

# Run preprocessing
python "$SCRIPT_DIR/preprocessing.py"