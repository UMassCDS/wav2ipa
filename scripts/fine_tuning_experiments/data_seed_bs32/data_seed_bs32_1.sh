#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=24GB
#SBATCH -p gpu-preempt
#SBATCH -G 4
#SBATCH --constraint=vram40
#SBATCH --nodes=1
#SBATCH --exclude gpu010,uri-gpu010
#SBATCH --time 3:00:00
#SBATCH -o %j_fine_tune_data_seed_bs32_1.out
#SBATCH --mail-type END

batch_size=4
grad_acc=4
learning_rate=3e-4
model_dir=data/models/fine_tune_data_seed_bs32_1
base_model=excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k

dataset_cache=dataset_cache
data_dir=data/buckeye


module load conda/latest
conda activate ./env_cuda124

python --version

pip list

multipa-train -bm $base_model \
    --output_dir "$model_dir" --data_dir "$data_dir" --cache_dir "$dataset_cache" \
    --use_gpu --num_train_epochs 10 --num_proc 8 \
    --learning_rate $learning_rate --per_device_train_batch_size $batch_size --gradient_accumulation_steps $grad_acc --mask_time_length 4 \
    --train_seed 358 \
    buckeye --train_samples 4000 --val_samples 5605
