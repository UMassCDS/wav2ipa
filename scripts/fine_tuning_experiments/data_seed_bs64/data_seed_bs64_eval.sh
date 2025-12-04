#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=12GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH --time 08:00:00
#SBATCH -o %j_timit_fine_tune_data_seed_bs64_eval.out
#SBATCH --mail-type END


# Evaluation results for our the models that change the data seed only 

EVAL_RESULTS_CSV=data/evaluation_results/aggregate_metrics/timit_fine_tune_data_seed_bs64_eval.csv
DETAILED_RESULTS_DIR=data/evaluation_results/detailed_predictions
EDIT_DIST_DIR=data/evaluation_results/edit_distances
DATA_DIR=data/buckeye
MODEL_DIR=data/models

module load conda/latest
conda activate ./env_cuda124

multipa-evaluate --local_models \
  $MODEL_DIR/fine_tune_hyperparam_tuning_1/wav2vec2-large-lv60_phoneme-timit_english_timit-4k-buckeye-ipa
  $MODEL_DIR/fine_tune_data_seed_bs64_1/wav2vec2-large-lv60_phoneme-timit_english_timit-4k-buckeye-ipa \
  $MODEL_DIR/fine_tune_data_seed_bs64_2/wav2vec2-large-lv60_phoneme-timit_english_timit-4k-buckeye-ipa \
  $MODEL_DIR/fine_tune_data_seed_bs64_3/wav2vec2-large-lv60_phoneme-timit_english_timit-4k-buckeye-ipa \
  $MODEL_DIR/fine_tune_data_seed_bs64_3/wav2vec2-large-lv60_phoneme-timit_english_timit-4k-buckeye-ipa \
 --eval_out $EVAL_RESULTS_CSV \
 --verbose_results_dir $DETAILED_RESULTS_DIR \
 --edit_dist_dir $EDIT_DIST_DIR \
 --no_space --data_dir $DATA_DIR \
 --use_gpu --num_proc 8
