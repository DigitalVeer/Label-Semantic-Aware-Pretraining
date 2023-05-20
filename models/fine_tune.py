import torch, os, json, argparse

from sys import argv as args
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset

import numpy as np
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def few_shot(model_name_or_path, train_set_path, test_set_path, val_set_path, output_dir, analysis_dir):
    prefix = "intent classification: "

    # creating directories:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # decoding labels
    def get_decoded_labels_preds(preds, labels):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        decoded_labels = [label for label in decoded_labels]
        return (decoded_preds, decoded_labels)

    def compute_basic_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds, decoded_labels = get_decoded_labels_preds(preds,labels)

        equals = 0
        for i in range(len(decoded_labels)): 
            if decoded_labels[i] == decoded_preds[i]:
                equals += 1
            
        return {"accuracy": equals/len(decoded_labels)}


    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(text_target=examples["intent"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_df = pd.read_csv(train_set_path)
    get_random_sample = lambda x: x.iloc[np.random.randint(0, len(x))]
    shots_list = [] # stores df for 1,2,4,8,16 shot training data
    prev = None

    for i in range(16):
        dummy = train_df.groupby('intent').apply(get_random_sample)
        curr = dummy.reset_index(drop = True)
        if i != 0:
            #curr = prev.concat(curr, ignore_index = True)
            curr = pd.concat([prev, curr], ignore_index=True)
        
        prev = curr
        
        if i == 0 or i == 1 or i == 3 or i == 7 or i == 15:
            shots_list.append(curr)


    one_shot_train_df = shots_list[0]
    two_shot_train_df = shots_list[1]
    four_shot_train_df = shots_list[2]
    eight_shot_train_df = shots_list[3]
    sixteen_shot_train_df = shots_list[4]
    train_set_name = ['one_shot', 'two_shot', 'four_shot', 'eight_shot', 'sixteen_shot', 'full_resource']

    all_shots = {}

    all_shots['one_shot'] = one_shot_train_df
    all_shots['two_shot'] = two_shot_train_df
    all_shots['four_shot'] = four_shot_train_df
    all_shots['eight_shot'] = eight_shot_train_df
    all_shots['sixteen_shot'] = sixteen_shot_train_df
    all_shots['full_resource'] = train_df
    data_files = {"test": test_set_path, "validation": val_set_path}

    raw_datasets = load_dataset("csv", data_files = data_files)

    for index in range(len(train_set_name)):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        model.to(device)
        raw_datasets["train"] = Dataset.from_pandas(all_shots[train_set_name[index]])
        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(raw_datasets["train"].column_names)

        datacollator = DataCollatorForSeq2Seq(tokenizer = tokenizer, return_tensors= "pt", padding=True)
        training_args = Seq2SeqTrainingArguments(evaluation_strategy = "epoch", output_dir= output_dir,
                                learning_rate= 1e-2,
                                per_device_train_batch_size=32,
                                per_device_eval_batch_size=32,
                                save_total_limit=3,
                                num_train_epochs=10,
                                predict_with_generate=True,
                                fp16= ( "cuda" in device ),
                                logging_strategy = 'epoch')
        trainer = Seq2SeqTrainer(model = model, 
                      args = training_args, 
                      train_dataset=tokenized_datasets["train"],
                      eval_dataset=tokenized_datasets["validation"],
                      data_collator = datacollator,
                      tokenizer = tokenizer,
                      compute_metrics = compute_basic_metrics
                      )
        trainer.train()
            # save model
        trainer.save_model()

        ## Evaluation on Test set.
        predictions, labels, metrics = trainer.predict(tokenized_datasets["test"])
        decoded_preds, decoded_labels = get_decoded_labels_preds(predictions,labels)
        print('For ' + model_name_or_path +  '_' + train_set_name[index] + ' metrics are: ')
        print(metrics)

        concat_preds = u'\n'.join(decoded_preds).encode('utf-8').strip()
        concat_labels = u'\n'.join(decoded_labels).encode('utf-8').strip()

        file_name_preds = os.path.join(analysis_dir, f'{train_set_name[index]}_predicted_labels.txt')
        file_name_labels = os.path.join(analysis_dir, f'{train_set_name[index]}_true_labels.txt')
        file_name_metrics = os.path.join(analysis_dir, f'{train_set_name[index]}_metrics.txt')

        with open(file_name_preds, 'wb') as f:
            f.write(concat_preds)

        print(f'Predicted labels are saved at:', Path(file_name_preds).absolute())

        with open(file_name_labels, 'wb') as f:
            f.write(concat_labels)

        print(f'True labels are saved at:', Path(file_name_labels).absolute())

        with open(file_name_metrics, "w") as json_file:
            json.dump(metrics, json_file)

        print('Metrics are saved at:', Path(file_name_metrics).absolute())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    
    parser.add_argument("--model_name_or_path", type=str, help="Name or path of the model")
    parser.add_argument("--train_set_path", type=str, help="Path to the training set")
    parser.add_argument("--test_set_path", type=str, help="Path to the test set")
    parser.add_argument("--val_set_path", type=str, help="Path to the validation set")
    parser.add_argument("--output_dir", type=str, help="Directory to output trained model")
    parser.add_argument("--analysis_dir", type=str, help="Directory for analysis")

    args = parser.parse_args()
    
    print("model_name_or_path", args.model_name_or_path)
    print("train_set_path", args.train_set_path)
    print("val_set_path", args.val_set_path)
    print("test_set_path", args.test_set_path)
    print("output_dir", args.output_dir)
    print("analysis_dir", args.analysis_dir)

    few_shot(args.model_name_or_path, args.train_set_path, args.test_set_path, args.val_set_path, args.output_dir, args.analysis_dir)