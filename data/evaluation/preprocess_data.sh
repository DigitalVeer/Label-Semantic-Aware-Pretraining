#!/bin/bash

# Retrieve the directory of the script
SCRIPT_DIR="$(dirname "$0")"

# Activate virtual environment
source "$SCRIPT_DIR/../../.venv/Scripts/activate"

python "$SCRIPT_DIR/ATIS/get_data.py"
python "$SCRIPT_DIR/SNIPS/get_data.py"
python "$SCRIPT_DIR/TOPS_Reminder/get_data.py"
python "$SCRIPT_DIR/TOPS_Weather/get_data.py"

# Delete preprocessed data
rm -rf "$SCRIPT_DIR/dataset/csv"
rm -rf "$SCRIPT_DIR/dataset/json"

# Run preprocessing
python "$SCRIPT_DIR/preprocessing.py"