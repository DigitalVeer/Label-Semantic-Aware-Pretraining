from transformers import Seq2SeqTrainingArguments
from transformers import set_seed
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq

from transformers import TrainerCallback
from transformers import Seq2SeqTrainer

import math
import pandas as pd
import numpy as np
import torch
import os
from sys import argv as args

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class AccCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer, data_args) -> None:
        super().__init__()
        self._trainer = trainer
        self.tokenizer = tokenizer
        self.data_args = data_args

    def on_epoch_end(self, args, state, control, **kwargs):
        tokenizer  = self.tokenizer
        data_args = self.data_args
        print("Calculating Accuracy on eval Dataset: START")
        model = self._trainer.model
        eval_dataloader = self._trainer.get_eval_dataloader()

        exact_match_acc = []
        first_word_acc = []
        for steps, inputs in enumerate(eval_dataloader):
            input_ids = inputs['input_ids'].to(device)
            labels = inputs['labels'].to(device)
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, max_new_tokens = data_args['max_target_length'])

            pred_labels = np.array([tokenizer.decode(masked, skip_special_tokens=True) for masked in outputs])
            gold_labels = np.array([tokenizer.decode(masked, skip_special_tokens=True) for masked in labels])
            
            # remove space before period/question mark
            gold_labels = np.array([word.replace(' ?', '?').replace(' .', '.').replace(' ,', ',') for word in gold_labels])
            
            # total masked tokens
            total_masked_labels = len(pred_labels)
            
            exact_match_acc.append(np.sum(pred_labels == gold_labels)/ total_masked_labels)
            
            # first word accuracy
            pred_labels = np.array([word.split()[0] if len(word.split()) else '' for word in pred_labels])
            gold_labels = np.array([word.split()[0] if len(word.split()) else '' for word in gold_labels])
            first_word_acc.append(np.sum(pred_labels == gold_labels)/ total_masked_labels)
        
        exact_match_acc_mean = np.mean(exact_match_acc)
        first_word_acc_mean = np.mean(first_word_acc)
            
        print(f"Epoch {state.epoch} Exact words match accuracy: {exact_match_acc_mean}")
        print(f"Epoch {state.epoch} First word match accuracy: {first_word_acc_mean}")
        print("Calculating Accuracy on eval Dataset: END")

class PerplexityCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        print("Calculating Perplexity on eval Dataset: START")
        model = self._trainer.model
        eval_dataloader = self._trainer.get_eval_dataloader()
        
        loss = []
        for steps, inputs in enumerate(eval_dataloader):
            input_ids = inputs['input_ids'].to(device)
            labels = inputs['labels'].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
            loss.append(outputs.loss.item())

        loss_mean = np.mean(loss)
        print(f"Epoch {state.epoch} loss_mean:", loss_mean)
        print(f"Epoch {state.epoch} perplexity:", np.exp(loss_mean))
        print("Calculating Perplexity on eval Dataset: END")
        
class PrintLossCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        history = state.log_history
        df = pd.DataFrame(history)
        
        # Filter rows for the last epoch
        last_epoch = df[df['epoch'] == df['epoch'].max()]

        # Select the columns of interest
        losses = last_epoch[['loss', 'eval_loss']]

        eval_loss = losses[['eval_loss']].dropna().iloc[0,0]
        train_loss = losses[['loss']].dropna().iloc[0,0]
        print(f"Epoch {state.epoch} Train loss:", train_loss)
        print(f"Epoch {state.epoch} Validation loss:", eval_loss)
        try:
            train_perplexity = math.exp(train_loss)
        except OverflowError:
            train_perplexity = math.inf
        try:
            eval_perplexity = math.exp(eval_loss)
        except OverflowError:
            eval_perplexity = math.inf
        print(f'Epoch {state.epoch} Train Perplexity:', train_perplexity)
        print(f'Epoch {state.epoch} Validation Perplexity:', eval_perplexity)

def preTrain(model_name_or_path, train_file, val_file, output_dir, is_random_weights = False, rand_weight_model_path = None):

    # Data related Arguments
    data_args = {
        'train_file': train_file,
        'validation_file': val_file,
        'max_target_length': 128,
        'max_source_length': 512,
        'ignore_pad_token_for_loss': True,
        }

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir,
        predict_with_generate= False,
        do_train= True,
        do_eval= True,
        per_device_train_batch_size= 16,
        per_device_eval_batch_size= 2,
        gradient_accumulation_steps= 4,
        learning_rate= 5e-3,
        evaluation_strategy= 'epoch',
        num_train_epochs= 5,
        save_total_limit= 2,
        save_strategy= 'epoch',
        load_best_model_at_end= True,
        logging_strategy='epoch',
        seed= 42
        )

    # set up the seed for persistent outputs
    set_seed(42)

    # create datasets
    print("Loading Datasets...")
    data_files = { 'train': data_args['train_file'], 'validation': data_args['validation_file'] }
    datasets = load_dataset('json', data_files=data_files)

    # Load config, tokenizer, model
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast = True)

    # Randomize model weights
    if is_random_weights:
        if rand_weight_model_path:
            model = AutoModelForSeq2SeqLM.from_pretrained(rand_weight_model_path)
        else:
            model = AutoModelForSeq2SeqLM.from_config(config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        config = config,
        from_tf=bool('.ckpt' in model_name_or_path)
    )

    # Tokenize input and target
    ## preprocess function
    def preprocess_function(examples):
        inputs = [ex for ex in examples['inputs']]
        targets = [ex for ex in examples['targets']]
        model_inputs = tokenizer(inputs, max_length= data_args['max_source_length'], padding = False, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length = data_args['max_target_length'], padding = False, truncation=True)

        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    train_dataset, eval_dataset = datasets['train'], datasets['validation']

    # logging train and eval dataset size
    print("train dataset", train_dataset)
    print("eval dataset", eval_dataset)

    # tokenize
    print("Tokenizing...")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=True,
    )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=True,
    )

    # create data_collator
    label_pad_token_id = -100 if data_args['ignore_pad_token_for_loss'] else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=label_pad_token_id)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Add callbacks to the model
    trainer.add_callback(PrintLossCallback())
    # trainer.add_callback(AccCallback(trainer, tokenizer, data_args))
    # trainer.add_callback(PerplexityCallback(trainer))

    # attach to available device
    model.to(device)

    # Train
    print("Training....")
    train_result = trainer.train()

    # save model
    trainer.save_model()

    # Save the trainer state explicitly
    output_train_file = os.path.join(training_args.output_dir, 'train_results.txt')
    if trainer.is_world_process_zero():
        with open(output_train_file, 'w') as writer:
            for key, value in sorted(train_result.metrics.items()):
                writer.write(f'{key} = {value}\n')

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(os.path.join(training_args.output_dir, 'trainer_state.json'))


if __name__ == "__main__":
    if len(args) < 6: 
        raise Exception("Please provide valid argument")
    
    model_name_or_path = args[1]
    train_file = args[2]
    val_file = args[3]
    output_dir = args[4]
    is_random_weights = True if args[5] == 'True' else False
    
    if is_random_weights and len(args) > 6:
        rand_weight_model_path = args[6]
    else:
        rand_weight_model_path = None

    print("model_name_or_path", model_name_or_path)
    print("train_file", train_file)
    print("val_file", val_file)
    print("output_dir", output_dir)
    print("is_random_weights",is_random_weights)
    print("rand_weight_model_path", rand_weight_model_path)

    preTrain(model_name_or_path, train_file, val_file, output_dir, is_random_weights, rand_weight_model_path)
    