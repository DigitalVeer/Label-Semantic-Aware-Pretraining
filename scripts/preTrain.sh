#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50GB  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o ./logs/slurm-%j.out  # %j = job ID

# load conda
module load conda

# activate vadops env
conda activate /work/pi_adrozdov_umass_edu/$USER/envs/vadops

python ../models/preTraining.py $1 $2 $3 $4 $5 $6

# Example
# Note: All trainings are done on batch_size = 16, lr = 5e-3
# Initial comparision: Which preTraining is better
# Label Denoising:
# Job: 7503186(batch_size = 8, lr = 5e-4), 7503816
# For: Gold + Silver + t5-small  (model_3)
# sbatch preTrain.sh t5-small ../data/pretraining/preprocessed_data/label_denoising/full_train.json ../data/pretraining/preprocessed_data/label_denoising/full_val.json ../output/label_denoising/gold_silver_model_3 False

# Intent classification PreTraining:
# Job: 7503846
# For: Gold + Silver + t5-small (model_3)
# sbatch preTrain.sh t5-small ../data/pretraining/preprocessed_data/intent_classification/full_train.json ../data/pretraining/preprocessed_data/intent_classification/full_val.json ../output/intent_classification/gold_silver_model_3 False

# Span Denoising:
## Job: 7510654
# For: Gold + Silver + t5-small (model_3)
# sbatch preTrain.sh t5-small ../data/pretraining/preprocessed_data/random_span_denoising/full_train.json ../data/pretraining/preprocessed_data/random_span_denoising/full_val.json ../output/span_denoising/gold_silver_model_3 False

# Once we choose label denoising as best, the models we built are:

# For: Gold + Silver + t5-small + random weights (model_5)
## Job: 7508790, 7510374(epoch 10)
# sbatch preTrain.sh t5-small ../data/pretraining/preprocessed_data/label_denoising/full_train.json ../data/pretraining/preprocessed_data/label_denoising/full_val.json ../output/label_denoising/gold_silver_model_5 True ../output/t5-small_random

## Job: 7504008
# For: Gold + Silver + t5-small + without_intent (model_2)
# sbatch preTrain.sh t5-small ../data/pretraining/preprocessed_data/random_denoising/full_train.json ../data/pretraining/preprocessed_data/random_denoising/full_val.json ../output/without_intent/gold_silver_model_2 False

## Job: 7508789, 7510373(epoch 10)
# For: Gold + Silver + t5-small + without_intent + random weights (model_4)
# sbatch preTrain.sh t5-small ../data/pretraining/preprocessed_data/random_denoising/full_train.json ../data/pretraining/preprocessed_data/random_denoising/full_val.json ../output/without_intent/gold_silver_model_4 True ../output/t5-small_random

# How the model is behaving when we inducing more data into preTraining data:

# Job: 7503795
# For: Gold + t5-small (model_3)
# sbatch preTrain.sh t5-small ../data/pretraining/preprocessed_data/label_denoising/polyai-bank_train.json ../data/pretraining/preprocessed_data/label_denoising/polyai-bank_val.json ../output/label_denoising/gold_model_3 False



