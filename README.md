# Label Semantic Aware Pretraining
In this project, we explore the effectiveness of LSAP on few-shot intent classification tasks. *in progress*
# Setup Project with Poetry
[Poetry](https://python-poetry.org/) is a more powerful version of pip w/ easier virtual environment management. I recommend using it but you can also just use Pip (see below). I've simplified the Poetry installation process here and should take less than 5 minutes to have the project running.

1. Install Poetry (Run in Powershell).
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
# (Alternative) Setup Project with Pip

Use pip to install the dependencies from the requirements.txt file.  
```
pip install -r requirements.txt
```

# Data
We utilized the below datasets for different parts of our process.

### Pretraining:
- [PolyAI Bank](https://huggingface.co/datasets/PolyAI/banking77)
- [WikiHow](https://github.com/zharry29/wikihow-intent)

### Evaluation:
- [SNIPS](https://paperswithcode.com/dataset/snips)
- [ATIS](https://github.com/yvchen/JointSLU/tree/master/data)

# Data Layout

The data that is used throughout our project is all stored under the data folder. The data is stored in the following format:
```
data
├───pretraining
│   ├───dataset (storage for final datasets)
│   ├───polyai-bank
│   │   └───get_data.ipynb
│   ├────wikihow
│   │   └───get_data.ipynb
│   └───create_dataset.ipynb (merges pretraining datasets into one)
├───evaluation
```

# Authors
- [VEER]()
- [PHANI]()
- [LOKESH]()
- [SANJANA]()
