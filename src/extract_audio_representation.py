# -*- coding: utf-8 -*-
# @Time    : 3/12/23 10:23 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_audio_representation.py

# extract un-pooled audio representations (last layer) of a dataset [num_sample, 512, 768], 512 corresponds to 512 patches of each 10 second audios
# with some modification, this code can also be used for visual representations extraction

import argparse
import os
import models
import dataloader as dataloader
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch import nn
from numpy import dot
from numpy.linalg import norm

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def save_audio_feat(audio_model, val_loader, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat = []
    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            audio_input, video_input = a_input.to(device), v_input.to(device)
            with autocast():
                audio_output, _ = audio_model.module.forward_feat(audio_input, video_input)
                print(audio_output.shape)
            audio_output = audio_output.to('cpu').detach()
            A_a_feat.append(audio_output)
    A_a_feat = torch.cat(A_a_feat)
    array = A_a_feat.numpy()
    # should be [num_sample, 512, 768], 512 corresponds to 512 patches of each 10 second audios
    print('The extracted feature shape:', array.shape)
    np.savez(save_path, array=array)

    # # load the saved feature
    # loaded_data = np.load('tensor_data.npz')
    # loaded_array = loaded_data['array']

def extract_audio_feat(model, data, audio_conf, label_csv, num_class, save_path, model_type='pretrain', batch_size=48):
    print(model)
    print(data)
    frame_use = 5
    # eval setting
    val_audio_conf = audio_conf
    val_audio_conf['frame_use'] = frame_use
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.data_val = data
    args.label_csv = label_csv
    args.exp_dir = './exp/dummy'
    args.loss_fn = torch.nn.BCELoss()
    val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf), batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)
    # cav-mae only been ssl pretrained
    if model_type == 'pretrain':
        audio_model = models.CAVMAE(modality_specific_depth=11)
    # cav-mae only been ssl pretrained + supervisedly finetuned
    elif model_type == 'finetune':
        audio_model = models.CAVMAEFT(label_dim=num_class, modality_specific_depth=11)
    sdA = torch.load(model, map_location=device)
    if isinstance(audio_model, torch.nn.DataParallel) == False:
        audio_model = torch.nn.DataParallel(audio_model)
    msg = audio_model.load_state_dict(sdA, strict=True)
    print(msg)
    audio_model.eval()
    save_audio_feat(audio_model, val_loader, save_path)

# as 20k balanced training data
data = '/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/audioset/audioset_20k_cleaned.json'

label_csv = '/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/audioset/class_labels_indices.csv'
dataset = 'audioset'
audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
              'mode': 'eval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224, 'frame_use': 5}

# # download the model from https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1
# # model only been pretrained (not finetuned) with batch size 256, i.e., cav_mae++
# model = './cav_mae_pretrain.pth'
# extract_audio_feat(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=527, save_path = './as_bal_pretrain.npz', model_type='pretrain', batch_size=100)

# download the model from https://www.dropbox.com/s/itfw7p0ueq7z9og/as_46.6.pth?dl=1
# model pretrained with multi-modal and finetuned on just audio of audioset
model = './as_46.6.pth'
extract_audio_feat(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=527, save_path = './as_bal_ft.npz', model_type='finetune', batch_size=100)