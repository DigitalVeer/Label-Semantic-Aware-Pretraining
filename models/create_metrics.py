import os
from sys import argv as args
import pandas as pd
from collections import defaultdict
import json

if __name__ == "__main__":
    if len(args) < 2: 
        raise Exception("Please provide valid argument")

    analysis_dir = args[1]
    dataset_dirs = ['ATIS', 'SNIPS', 'TOPS_Reminder', 'TOPS_Weather']

    acc = defaultdict(list)
    first_word_acc = defaultdict(list)
    bleu_score = defaultdict(list)
    cosine_similarity = defaultdict(list)
    jaccard_score = defaultdict(list)

    for dir in dataset_dirs:
        acc = defaultdict(list)
        first_word_acc = defaultdict(list)
        bleu_score = defaultdict(list)
        cosine_similarity = defaultdict(list)
        jaccard_score = defaultdict(list)
        model_exists = []
        for i in range(1, 11):
          dataset_dir = os.path.join(analysis_dir, dir)
          fold = os.path.join(dataset_dir, f'model_{i}')
          if os.path.exists(fold):
            file_path = os.path.join(fold, 'results.json')
            print(file_path)
            with open(file_path, 'r') as file:
                data = json.load(file)
                model_exists.append(True)
          else:
                data = {}
                model_exists.append(False)
          
          for shot in ['one_shot', 'two_shot', 'four_shot', 'eight_shot', 'sixteen_shot', 'full_resource']:
              if shot not in data:
                acc[shot].append(0)
                first_word_acc[shot].append(0)
                bleu_score[shot].append(0)
                cosine_similarity[shot].append(0)
                jaccard_score[shot].append(0)
              else:
                metrics = data[shot]['results']
                acc[shot].append(metrics['accuracy']['exact_match'])
                first_word_acc[shot].append(metrics['accuracy']['first_word'])
                bleu_score[shot].append(metrics['bleu_score'])
                cosine_similarity[shot].append(metrics['cosine_similarity'])
                jaccard_score[shot].append(metrics['jaccard_score'])
          
        acc['model'] = model_exists
        first_word_acc['model'] = model_exists
        bleu_score['model'] = model_exists
        cosine_similarity['model'] = model_exists
        jaccard_score['model'] = model_exists

        pd.DataFrame(acc).to_csv(dataset_dir + '/acc.csv', index=False)
        pd.DataFrame(first_word_acc).to_csv(dataset_dir + '/first_word_acc.csv', index=False)
        pd.DataFrame(bleu_score).to_csv(dataset_dir + '/bleu_score.csv', index=False)
        pd.DataFrame(cosine_similarity).to_csv(dataset_dir + '/cosine_similarity.csv', index=False)
        pd.DataFrame(jaccard_score).to_csv(dataset_dir + '/jaccard_score.csv', index=False)