#!/bin/bash
#SBATCH -p a5
#SBATCH --qos regular
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="vgg-pretrain"
#SBATCH --output=../log/%j_vgg_pretrain.txt

# run cav-mae pretraining

set -x
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/avbyol/venv-a5/bin/activate
export TORCH_HOME=../../pretrained_models

model=cav-mae #  avmaejointpsymc11, avmaejointparss11
masking_ratio=0.75
mask_mode=unstructured
contrast_loss_weight=0.01
mae_loss_weight=1.0
norm_pix_loss=True
tr_pos=False

# you can replace with any checkpoint you want, but here we do another round of pretraining based on cav-mae-scale++ trained on AS-2M
cur_dir=$(pwd)
wget -nc https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1 -O cav-mae-scale++.pth
pretrain_path=${cur_dir}/cav-mae-scale++.pth

lr=5e-5
epoch=10
bal=None
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

exp_dir=./exp/testmae02-${dataset}-${model}-bal${bal}-lr${lr}-epoch${epoch}-bs${batch_size}-norm${norm_pix_loss}-c${contrast_loss_weight}-p${mae_loss_weight}-tp${tr_pos}-mr-${mask_mode}-${masking_ratio}-as-a5
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