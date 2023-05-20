source ../.venv/Scripts/activate

python ../models/fine_tune.py \
    --model_name_or_path 't5-small' \
    --train_set_path ../data/evaluation/dataset/csv/ATIS/ATIS_train.csv \
    --test_set_path ../data/evaluation/dataset/csv/ATIS/ATIS_test.csv  \
    --val_set_path  ../data/evaluation/dataset/csv/ATIS/ATIS_val.csv  \
    --output_dir  ../output/fine-tune_models/ATIS/model_1 \
    --analysis_dir ./analysis/ATIS/model_1

deactivate

# # model 1
# python fine_tune.py \
#     --model_name_or_path t5-small \
#     --train_set_path ../data/evaluation/ATIS/data/train.csv \
#     --val_set_path ../data/evaluation/ATIS/data/val.csv \
#     --test_set_path ../data/evaluation/ATIS/data/test.csv \
#     --output_dir ../output/fine-tune_models/ATIS/model_1 \
#     --analysis_dir ../analysis/ATIS/model_1

# # model 2
# python fine_tune.py \
#     --model_name_or_path ../output/without_intent/gold_silver_model_2 \
#     --train_set_path ../data/evaluation/ATIS/data/train.csv \
#     --val_set_path ../data/evaluation/ATIS/data/val.csv \
#     --test_set_path ../data/evaluation/ATIS/data/test.csv \
#     --output_dir ../output/fine-tune_models/ATIS/model_2 \
#     --analysis_dir ../analysis/ATIS/model_2

# # model 3
# python fine_tune.py \
#     --model_name_or_path ../output/label_denoising/gold_silver_model_3_1 \
#     --train_set_path ../data/evaluation/ATIS/data/train.csv \
#     --val_set_path ../data/evaluation/ATIS/data/val.csv \
#     --test_set_path ../data/evaluation/ATIS/data/test.csv \
#     --output_dir ../output/fine-tune_models/ATIS/model_3 \
#     --analysis_dir ../analysis/ATIS/model_3

# # model 4
# python fine_tune.py \
#     --model_name_or_path ../output/without_intent/gold_silver_model_4 \
#     --train_set_path ../data/evaluation/ATIS/data/train.csv \
#     --val_set_path ../data/evaluation/ATIS/data/val.csv \
#     --test_set_path ../data/evaluation/ATIS/data/test.csv \
#     --output_dir ../output/fine-tune_models/ATIS/model_4 \
#     --analysis_dir ../analysis/ATIS/model_4

# # model 5
# python fine_tune.py \
#     --model_name_or_path ../output/label_denoising/gold_silver_model_5 \
#     --train_set_path ../data/evaluation/ATIS/data/train.csv \
#     --val_set_path ../data/evaluation/ATIS/data/val.csv \
#     --test_set_path ../data/evaluation/ATIS/data/test.csv \
#     --output_dir ../output/fine-tune_models/ATIS/model_5 \
#     --analysis_dir ../analysis/ATIS/model_5

# # model 6
# python fine_tune.py \
#     --model_name_or_path ../output/intent_classification/gold_silver_model_3 \
#     --train_set_path ../data/evaluation/ATIS/data/train.csv \
#     --val_set_path ../data/evaluation/ATIS/data/val.csv \
#     --test_set_path ../data/evaluation/ATIS/data/test.csv \
#     --output_dir ../output/fine-tune_models/ATIS/model_6 \
#     --analysis_dir ../analysis/ATIS/model_6

# # model 7
# python fine_tune.py \
#     --model_name_or_path ../output/span_denoising/gold_silver_model_3 \
#     --train_set_path ../data/evaluation/ATIS/data/train.csv \
#     --val_set_path ../data/evaluation/ATIS/data/val.csv \
#     --test_set_path ../data/evaluation/ATIS/data/test.csv \
#     --output_dir ../output/fine-tune_models/ATIS/model_7 \
#     --analysis_dir ../analysis/ATIS/model_7

# # model 8
# python fine_tune.py \
#     --model_name_or_path ../output/label_denoising/gold_model_3 \
#     --train_set_path ../data/evaluation/ATIS/data/train.csv \
#     --val_set_path ../data/evaluation/ATIS/data/val.csv \
#     --test_set_path ../data/evaluation/ATIS/data/test.csv \
#     --output_dir ../output/fine-tune_models/ATIS/model_8 \
#     --analysis_dir ../analysis/ATIS/model_8

# # model 9
# python fine_tune.py \
#     --model_name_or_path ../output/without_intent/gold_silver_model_4_epoch_10 \
#     --train_set_path ../data/evaluation/ATIS/data/train.csv \
#     --val_set_path ../data/evaluation/ATIS/data/val.csv \
#     --test_set_path ../data/evaluation/ATIS/data/test.csv \
#     --output_dir ../output/fine-tune_models/ATIS/model_9 \
#     --analysis_dir ../analysis/ATIS/model_9

# # model 10
# python fine_tune.py \
#     --model_name_or_path ../output/label_denoising/gold_silver_model_5_epoch_10 \
#     --train_set_path ../data/evaluation/ATIS/data/train.csv \
#     --val_set_path ../data/evaluation/ATIS/data/val.csv \
#     --test_set_path ../data/evaluation/ATIS/data/test.csv \
#     --output_dir ../output/fine-tune_models/ATIS/model_10 \
#     --analysis_dir ../analysis/ATIS/model_10