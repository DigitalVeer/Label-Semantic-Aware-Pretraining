#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50GB  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 2:00:00  # Job time limit
#SBATCH -o ./logs/slurm-%j.out  # %j = job ID

# load conda
module load conda

# activate vadops env
conda activate /work/pi_adrozdov_umass_edu/$USER/envs/vadops

python ../models/fine_tune.py $1 $2 $3 $4 $5 $6

# model 1
# sbatch fine_tune.sh t5-small ../data/evaluation/ATIS/data/train.csv ../data/evaluation/ATIS/data/val.csv ../data/evaluation/ATIS/data/test.csv ../output/fine-tune_models/ATIS/model_1 ../analysis/ATIS/model_1

# model 2
# sbatch fine_tune.sh ../output/without_intent/gold_silver_model_2 ../data/evaluation/ATIS/data/train.csv ../data/evaluation/ATIS/data/val.csv ../data/evaluation/ATIS/data/test.csv ../output/fine-tune_models/ATIS/model_2 ../analysis/ATIS/model_2

# model 3
# sbatch fine_tune.sh ../output/label_denoising/gold_silver_model_3_1 ../data/evaluation/ATIS/data/train.csv ../data/evaluation/ATIS/data/val.csv ../data/evaluation/ATIS/data/test.csv ../output/fine-tune_models/ATIS/model_3 ../analysis/ATIS/model_3

# model 4
# sbatch fine_tune.sh ../output/without_intent/gold_silver_model_4 ../data/evaluation/ATIS/data/train.csv ../data/evaluation/ATIS/data/val.csv ../data/evaluation/ATIS/data/test.csv ../output/fine-tune_models/ATIS/model_4 ../analysis/ATIS/model_4

# model 5
# sbatch fine_tune.sh ../output/label_denoising/gold_silver_model_5 ../data/evaluation/ATIS/data/train.csv ../data/evaluation/ATIS/data/val.csv ../data/evaluation/ATIS/data/test.csv ../output/fine-tune_models/ATIS/model_5 ../analysis/ATIS/model_5

# model 6
# sbatch fine_tune.sh ../output/intent_classification/gold_silver_model_3 ../data/evaluation/ATIS/data/train.csv ../data/evaluation/ATIS/data/val.csv ../data/evaluation/ATIS/data/test.csv ../output/fine-tune_models/ATIS/model_6 ../analysis/ATIS/model_6

# model 7
# sbatch fine_tune.sh ../output/span_denoising/gold_silver_model_3 ../data/evaluation/ATIS/data/train.csv ../data/evaluation/ATIS/data/val.csv ../data/evaluation/ATIS/data/test.csv ../output/fine-tune_models/ATIS/model_7 ../analysis/ATIS/model_7

# model 8
# sbatch fine_tune.sh ../output/label_denoising/gold_model_3 ../data/evaluation/ATIS/data/train.csv ../data/evaluation/ATIS/data/val.csv ../data/evaluation/ATIS/data/test.csv ../output/fine-tune_models/ATIS/model_8 ../analysis/ATIS/model_8

# model 9
# sbatch fine_tune.sh ../output/without_intent/gold_silver_model_4_epoch_10 ../data/evaluation/ATIS/data/train.csv ../data/evaluation/ATIS/data/val.csv ../data/evaluation/ATIS/data/test.csv ../output/fine-tune_models/ATIS/model_9 ../analysis/ATIS/model_9

# model 10
# sbatch fine_tune.sh ../output/label_denoising/gold_silver_model_5_epoch_10 ../data/evaluation/ATIS/data/train.csv ../data/evaluation/ATIS/data/val.csv ../data/evaluation/ATIS/data/test.csv ../output/fine-tune_models/ATIS/model_10 ../analysis/ATIS/model_10







