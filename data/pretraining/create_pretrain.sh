#activate the environment in .venv
source .venv/bin/activate

python polyai-bank/get_data.py
python wikihow/get_data.py

#delete preprocessed data
rm -rf preprocessed_data
rm -rf dataset/csv
rm -rf dataset/json

python preprocessing.py