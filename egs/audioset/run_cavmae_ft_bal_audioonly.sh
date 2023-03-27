#!/bin/bash
##SBATCH -p a5
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[1,2],sls-sm-[2,5,6,7,12]
#SBATCH -p gpu
##SBATCH --qos regular
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="as-bal-ft"
#SBATCH --output=../log/%j_as_ft.txt

# pretrained with audio-visual data, but finetuned only for audio

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/avbyol/venv-a5/bin/activate
export TORCH_HOME=../../pretrained_models

model=cav-mae-ft
ftmode=audioonly

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)
wget -nc https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1 -O cav-mae-scale++.pth
pretrain_path=${cur_dir}/cav-mae-scale++.pth

freeze_base=False
head_lr=100 # newly initialized ft layers uses 100 times larger than the base lr

bal=None
lr=5e-5
epoch=15
lrscheduler_start=5
lrscheduler_decay=0.5
lrscheduler_step=1
wa=True
wa_start=3
wa_end=15
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=36
label_smooth=0.1

dataset=audioset
tr_data=/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/audioset/audioset_20k_cleaned.json
te_data=/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/audioset/audioset_eval_cleaned.json
label_csv=/data/sls/scratch/yuangong/convast/egs/audioset/data/class_labels_indices.csv

exp_dir=./exp/testmae06-bal-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-a5
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_cavmae_ft.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss BCE --metrics mAP --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--num-workers 32 --skip_frame_agg True