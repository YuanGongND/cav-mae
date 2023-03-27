#!/bin/bash
#SBATCH -p a5
#SBATCH --qos regular
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="vgg-ft"
#SBATCH --output=../log/%j_vgg_ft.txt

# finetune cav-mae pretrained on AS-2M with VGGSound dataset
# you can change pretrain_path to other cav-mae models

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/avbyol/venv-a5/bin/activate
export TORCH_HOME=../../pretrained_models

model=cav-mae-ft
ftmode=multimodal

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)
wget -nc https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1 -O cav-mae-scale++.pth
pretrain_path=${cur_dir}/cav-mae-scale++.pth

freeze_base=False
head_lr=10 # newly initialized ft layers uses 10 times larger than the base lr

bal=bal
lr=1e-4
epoch=10
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa=True
wa_start=3
wa_end=10
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=48
label_smooth=0.1

dataset=vggsound
tr_data=/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/vggsound/vgg_train_cleaned.json
te_data=/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/vggsound/vgg_test_cleaned.json

exp_dir=./exp/testmae02-${dataset}-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-a5
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_cavmae_ft.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv /data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/vggsound/class_labels_indices_vgg.csv --n_class 309 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss CE --metrics acc --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--num-workers 32