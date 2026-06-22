<div align="center">

# Label Semantic Aware Pretraining
  
  
In this research project, we explore the effectiveness of LSAP on few-shot intent classification tasks. Our aim is to implement the LSAP technique on a series of T5-small models and evaluate their performance across diverse few-shot settings. The original Label Semantic Aware Pre-training paper can be found [here](https://arxiv.org/pdf/2204.07128.pdf).

 # Setup
  
</div>

This project is managed using [Poetry](https://python-poetry.org/), an alternative to pip with virtual environment management. It targets **Python 3.10–3.12** and the modern ML stack (Transformers ≥ 4.46, PyTorch ≥ 2.2, Datasets ≥ 3.0).

1. Install Poetry.
```bash
# Linux / macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
2. Make sure Poetry is on your `PATH` (the installer prints the location; on Windows it is typically `%APPDATA%\pypoetry\venv\Scripts`).
3. Install dependencies in the project directory:
```bash
poetry config virtualenvs.in-project true   # creates ./.venv (used by the scripts)
poetry install
```
4. Run commands inside the environment with `poetry run`, e.g.:
```bash
poetry run python models/fine_tune.py --help
```
(Poetry 2.x removed the built-in `poetry shell`; use `poetry run`, or `poetry env activate` to print an activation command.)

## Pip Setup (Alternative)

Assuming you have pip configured, install the dependencies directly:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> For a CUDA build of PyTorch, install `torch` from the official index first
> (see the comment at the top of `requirements.txt`).

<div align="center">

# How To Run
  
</div>

To generate data from scratch:
```
cd scripts
sh generate_data.sh
```

To pretrain models (requires configuration based on environment):
```
cd scripts
sh do_pretrain.sh
```
The training arguments can be changed inside `do_pretrain.sh` to replicate the different models attempted in our paper.

To fine-tune models:
```
cd scripts
sh fine_tuning.sh
```

### Choosing a base model

Both `models/pretrain.py` and `models/fine_tune.py` take a
`--model_name_or_path`, so any Hugging Face seq2seq checkpoint works as a
drop-in replacement for `t5-small`. Two useful upgrades:

- **Larger T5:** `t5-base`, `t5-large` — closer to the scale used in the
  original LSAP paper (the much smaller `t5-small` is the main reason our
  accuracies trail theirs).
- **Instruction-tuned, same architecture:** `google/flan-t5-small`,
  `google/flan-t5-base` — drop-in replacements with stronger few-shot
  behavior out of the box.

Example:
```
python models/fine_tune.py --model_name_or_path google/flan-t5-base ...
```

<div align="center">

# Data

</div>

Our project relies on a variety of datasets, each playing a key role in different stages, and we provide a concise overview of their significance.

### Pretraining

1. **PolyAI Bank:** The [PolyAI Bank](https://huggingface.co/datasets/PolyAI/banking77) dataset contains banking-related utterances. This dataset serves a large amount of customer intents and is available via the Hugging Face library.

2. **WikiHow:** The [WikiHow](https://github.com/zharry29/wikihow-intent) dataset is sourced from the WikiHow website. It pairs the longest step in a WikiHow article with the article title (sans "How To") as its intent. 

### Evaluation

1. **SNIPS:** We use the [SNIPS](https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines) dataset as it is a popular benchmark in intent classification tasks.

2. **ATIS:**  [ATIS](https://github.com/yvchen/JointSLU/tree/master/data) (Airline Travel Information System) houses user queries concerning flight reservations, schedules, and other travel-related subjects. Similar to the authors, we use this to evaluate intent classification.

3. **TOPv2:** Finally, the [TOPv2](https://fb.me/TOPv2Dataset) dataset developed by Facebook AI encompasses user queries across various domains, including reminders and weather. We use a focus on TOPv2Weather and TOPv2Reminder for this project, as the original authors.


### Generation
To generate the pretraining data, run the following script:
```bash
sh data/pretraining/preprocess_data.sh
```

<div align="center">

# Data Layout

</div>

The data that is used throughout our project is all stored under the data folder. The data is stored in the following format:
```
data
├───pretraining
│   ├───dataset (storage for raw data)
│   │───preprocessed_data (stores tokenized data)
│   ├───polyai-bank
│   │   └───get_data.py (data generator in each dataset)
│   ├────wikihow
│   │   └───get_data.py 
│   preprocessing.py (tokenizes raw dataset & stores them in preprocessed_data)
├───evaluation
│   ├───atis
│   ├───snips
│   ├───tops_reminder
│   ├───tops_weather
│   │───dataset (storage for raw data)
│   preprocessing.py (stores datasets into dataset folder)
```
<div align="center">

# :busts_in_silhouette: Authors

</div>

- [**Veer Singh**](https://github.com/DigitalVeer)
- [**Phanindra PVS**](https://github.com/PVSPHANINDRA)
- [**Lokesh Tangella**](https://github.com/lokesh9920)
- [**Sanjana Radhakrishna**]()
