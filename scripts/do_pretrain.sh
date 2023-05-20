source ../.venv/Scripts/activate

python ../models/pretrain.py \
    --model_name_or_path 't5-small' \
    --train_file ../data/pretraining/preprocessed_data/label_denoising/full_train.json \
    --val_file ../data/pretraining/preprocessed_data/label_denoising/full_val.json \
    --output_dir  ../output/label_denoising/gold_silver_model_3 \
    --is_random_weights False \
    --rand_weight_model_path None

deactivate

# Example
# Note: All trainings are done on batch_size = 16, lr = 5e-3
# Initial comparision: Which preTraining is better
# Label Denoising:
# Job: 7503186(batch_size = 8, lr = 5e-4), 7503816
# For: Gold + Silver + t5-small  (model_3)
# python pretraining.py \
#     --model_name_or_path t5-small \
#     --train_file ../data/pretraining/preprocessed_data/label_denoising/full_train.json \
#     --val_file ../data/pretraining/preprocessed_data/label_denoising/full_val.json \
#     --output_dir ../output/label_denoising/gold_silver_model_3 \
#     --is_random_weights False \
#     --rand_weight_model_path None

# # Intent classification PreTraining:
# # Job: 7503846
# # For: Gold + Silver + t5-small (model_3)
# python pretraining.py \
#     --model_name_or_path t5-small \
#     --train_file ../data/pretraining/preprocessed_data/intent_classification/full_train.json \
#     --val_file ../data/pretraining/preprocessed_data/intent_classification/full_val.json \
#     --output_dir ../output/intent_classification/gold_silver_model_3 \
#     --is_random_weights False \
#     --rand_weight_model_path None

# # Span Denoising:
# # Job: 7510654
# # For: Gold + Silver + t5-small (model_3)
# python pretraining.py \
#     --model_name_or_path t5-small \
#     --train_file ../data/pretraining/preprocessed_data/random_span_denoising/full_train.json \
#     --val_file ../data/pretraining/preprocessed_data/random_span_denoising/full_val.json \
#     --output_dir ../output/span_denoising/gold_silver_model_3 \
#     --is_random_weights False \
#     --rand_weight_model_path None

# # Once we choose label denoising as best, the models we built are:

# # For: Gold + Silver + t5-small + random weights (model_5)
# # Job: 7508790, 7510374(epoch 10)
# python pretraining.py \
#     --model_name_or_path t5-small \
#     --train_file ../data/pretraining/preprocessed_data/label_denoising/full_train.json \
#     --val_file ../data/pretraining/preprocessed_data/label_denoising/full_val.json \
#     --output_dir ../output/label_denoising/gold_silver_model_5 \
#     --is_random_weights True \
#     --rand_weight_model_path ../output/t5-small_random

# # Job: 7504008
# # For: Gold + Silver + t5-small + without_intent (model_2)
# python pretraining.py \
#     --model_name_or_path t5-small \
#     --train_file ../data/pretraining/preprocessed_data/random_denoising/full_train.json \
#     --val_file ../data/pretraining/preprocessed_data/random_denoising/full_val.json \
#     --output_dir ../output/without_intent/gold_silver_model_2 \
#     --is_random_weights False \
#     --rand_weight_model_path None

# # Job: 7508789, 7510373(epoch 10)
# # For: Gold + Silver + t5-small + without_intent + random weights (model_4)
# python pretraining.py \
#     --model_name_or_path t5-small \
#     --train_file ../data/pretraining/preprocessed_data/random_denoising/full_train.json \
#     --val_file ../data/pretraining/preprocessed_data/random_denoising/full_val.json \
#     --output_dir ../output/without_intent/gold_silver_model_4 \
#     --is_random_weights True \
#     --rand_weight_model_path ../output/t5-small_random

# # How the model is behaving when we inducing more data into preTraining data:

# # Job: 7503795
# # For: Gold + t5-small (model_3)
# python pretraining.py \
#     --model_name_or_path t5-small \
#     --train_file ../data/pretraining/preprocessed_data/label_denoising/polyai-bank_train.json \
#     --val_file ../data/pretraining/preprocessed_data/label_denoising/polyai-bank_val.json \
#     --output_dir ../output/label_denoising/gold_model_3 \
#     --is_random_weights False \
#     --rand_weight_model_path None
