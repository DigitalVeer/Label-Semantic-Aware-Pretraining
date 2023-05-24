<div align="center">

# Label Semantic Aware Pretraining
  
  
In this research project, we evaluate we explore the effectiveness of LSAP on few-shot intent classification tasks. Our principal aim is to implement the LSAP technique on a series of T5-small models and evaluate their performance across diverse few-shot settings, comparing it to baseline models. The original Label Semantic Aware Pre-training paper can be found [here](https://arxiv.org/pdf/2204.07128.pdf).

 # Setup
  
</div>

This project is managed using [Poetry](https://python-poetry.org/), an alternative to pip with virtual environment management.

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
## Pip Setup (Altnerative)

Assuming you have pip installed and configured on your system, you can use pip to install the dependencies.  
```
pip install -r requirements.txt
```

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
sh pretrain.sh
```
The training arguments can be changed inside the pretrain.sh to replicate different models attempted in our paper.

To fine-tune models:
```
cd scripts
sh fine-tune.sh
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
