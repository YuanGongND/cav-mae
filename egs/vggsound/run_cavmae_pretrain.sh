#!/bin/bash
#SBATCH -p a5
#SBATCH --qos regular
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="vgg-pretrain"
#SBATCH --output=../log/%j_vgg_pretrain.txt

# run cav-mae pretraining from original mae checkpoint
set -x
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/avbyol/venv-a5/bin/activate
export TORCH_HOME=../../pretrained_models

model=cav-mae
masking_ratio=0.75
mask_mode=unstructured
contrast_loss_weight=0.01
mae_loss_weight=1.0
tr_pos=False
norm_pix_loss=True

# you can use any checkpoints with a decoder, but by default, we use vision-MAE checkpoint
cur_dir=$(pwd)
wget -nc https://www.dropbox.com/s/9nlz523a5q52w86/ori_mae_11.pth?dl=1 -O IN-initial.pth
pretrain_path=${cur_dir}/IN-initial.pth

bal=None
lr=1e-4
epoch=25
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
mixup=0.0
batch_size=120
lr_adapt=False

dataset=vggsound
tr_data=/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/vggsound/vgg_train_cleaned.json
te_data=/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/vggsound/vgg_test_cleaned.json

exp_dir=./exp/testmae02-${dataset}-${model}-bal${bal}-lr${lr}-epoch${epoch}-bs${batch_size}-norm${norm_pix_loss}-c${contrast_loss_weight}-p${mae_loss_weight}-tp${tr_pos}-mr-${mask_mode}-${masking_ratio}-a5
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_cavmae_pretrain.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv /data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/vggsound/class_labels_indices_vgg.csv --n_class 309 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--mixup ${mixup} --bal ${bal} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} --warmup True \
--lr_adapt ${lr_adapt} \
--norm_pix_loss ${norm_pix_loss} \
--pretrain_path ${pretrain_path} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--tr_pos ${tr_pos} --masking_ratio ${masking_ratio} --mask_mode ${mask_mode}