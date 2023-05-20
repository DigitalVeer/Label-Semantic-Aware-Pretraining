# Label Semantic Aware Pretraining
In this research project, we evaluate we explore the effectiveness of LSAP on few-shot intent classification tasks. Our principal aim is to implement the LSAP technique on a series of T5-small models and evaluate their performance across diverse few-shot settings, comparing it to baseline models. The original Label Semantic Aware Pre-training paper can be found [here](https://arxiv.org/pdf/2204.07128.pdf).

## Poetry Setup

[Poetry](https://python-poetry.org/) is a more powerful version of pip w/ easier virtual environment management. I recommend using it but you can also just use Pip (see below). I've simplified the Poetry installation process here and should take less than 5 minutes to have the project running.

1. Install Poetry (Powershell).
```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
2. Add Poetry to PATH:
```
C:\Users\NAME\AppData\Roaming\pypoetry\venv\Scripts
```
3. Run in project directory:  
```
poetry config virtualenvs.in-project true
poetry install
```
4. To activate the virtual environment, run:  
```
poetry shell
```
To deactivate the virtual environment, run:  
```
exit
```
## Pip Setup

Assuming you have pip installed and configured on your system alreayd, you can use pip to install the dependencies from the requirements.txt file.  
```
pip install -r requirements.txt
```

# How to Run

To generate data from scratch:
```
cd scripts
sh generate_data.sh
```

To pretrain models (requires configuration based on environment):
```
cd scripts
sh pretrain.sh
```
The training arguments can be changed inside the pretrain.sh to replicate different models attempted in our paper.

To fine-tune models:
```
cd scripts
sh fine-tune.sh
```

# Data
We utilized the below datasets for different parts of our process. 

### Pretraining:
- [PolyAI Bank](https://huggingface.co/datasets/PolyAI/banking77)
- [WikiHow](https://github.com/zharry29/wikihow-intent)

### Evaluation:
- [SNIPS](https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines)
- [ATIS](https://github.com/yvchen/JointSLU/tree/master/data)
- [TOPv2](https://fb.me/TOPv2Dataset)

To generate the pretraining data, run:
```
sh data/pretraining/preprocess_data.sh
```

# Data Layout

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

# Authors
- [Veer Singh](https://github.com/DigitalVeer)
- [Phanindra PVS](https://github.com/PVSPHANINDRA)
- [Lokesh Tangella](https://github.com/lokesh9920)
- [Sanjana Radhakrishna]()
