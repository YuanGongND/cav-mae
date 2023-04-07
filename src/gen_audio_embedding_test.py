# -*- coding: utf-8 -*-
# @Time    : 3/12/23 10:23 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : retrieval.py

# extract audio or visual embedding from a finetuned CAVMAEFT model

import json
import argparse
import os
import models
import dataloader as dataloader
import torch
import numpy as np
from torch.cuda.amp import autocast

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# generate audio embedding
def gen_audio_embedding(audio_model, val_loader, audio_rep_order='temporal', save_name_list=None):
    # initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = audio_model.to(device)
    audio_model.eval()

    save_idx = 0
    with torch.no_grad():
        for i, (a_input, _, labels) in enumerate(val_loader):
            audio_input = a_input.to(device)
            with autocast():
                # audio only, visual input is None
                audio_output = audio_model.module.forward_feat(audio_input, None, mode='a')
                #print('audio output shape', audio_output.shape) # 512 = 8 * 64, in shape [f, t]
                if audio_rep_order == 'temporal':
                    assert audio_output.shape[1] == 8*64
                    audio_output = audio_output.reshape(audio_output.shape[0], 8, 64, audio_output.shape[-1])
                    audio_output = torch.mean(audio_output, dim=1) # mean pool over the frequency dimension
                    #print('audio output shape', audio_output.shape)
                # normalization
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)

            audio_output = audio_output.to('cpu').detach()
            for cur_idx in range(audio_output.shape[0]):
                print(audio_output[cur_idx][:5])
                exit()
                np.savez_compressed(save_name_list[save_idx], audio_output[cur_idx])
                #print(audio_output[cur_idx].shape)
                save_idx += 1
                if save_idx % 100 == 0:
                    print('processed {:d} samples.'.format(save_idx))
    print(save_idx)

def batch_gen_audio_embedding(model, data, audio_conf, label_csv, num_class, model_type='finetune', batch_size=48, save_name_list=None):
    print(model)
    print(data)
    # eval setting
    val_audio_conf = audio_conf
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.data_val = data
    args.label_csv = label_csv
    args.exp_dir = './exp/dummy'
    args.loss_fn = torch.nn.BCELoss()
    cur_dataset = dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf)
    assert len(cur_dataset) == len(save_name_list)
    print('dataset has ', len(cur_dataset))
    val_loader = torch.utils.data.DataLoader(cur_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)
    # cav-mae only been ssl pretrained
    if model_type == 'pretrain':
        audio_model = models.CAVMAE(modality_specific_depth=11)
    # cav-mae only been ssl pretrained + supervisedly finetuned
    elif model_type == 'finetune':
        audio_model = models.CAVMAEFT(label_dim=num_class, modality_specific_depth=11)
    elif model_type == 'finetune_audio':
        audio_model = models.CAVMAEFTAudio(label_dim=num_class, modality_specific_depth=11)
    sdA = torch.load(model, map_location=device)
    if isinstance(audio_model, torch.nn.DataParallel) == False:
        audio_model = torch.nn.DataParallel(audio_model)
    msg = audio_model.load_state_dict(sdA, strict=False)
    print(msg)
    audio_model.eval()
    gen_audio_embedding(audio_model, val_loader, audio_rep_order='temporal', save_name_list=save_name_list)

model = '/data/sls/scratch/yuangong/cav-mae/egs/audioset/exp/testmae01-full-cav-mae-ft-1e-5-2-0.5-1-bs48-ldaFalse-audioonly-fzFalse-h50-a5/models/best_audio_model.pth'
res = []

#data = '/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/audioset/audioset_eval_cleaned.json'
#data = '/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/audioset/audioset_2m_cleaned.json'
data = '/data/sls/scratch/yuangong/audiollm/data/datafiles/audioset_sample.json'
label_csv = '/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/audioset/class_labels_indices.csv'
dataset = 'audioset'
audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
              'mode': 'eval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224, 'frame_use': 5}

with open(data, 'r') as fp:
    data_json = json.load(fp)
    data_json = data_json['data']
save_name_list = ['/data/sls/scratch/yuangong/audiollm/data/sample_output/' + x['video_id'] for x in data_json]

batch_gen_audio_embedding(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=527, model_type='finetune_audio', batch_size=600, save_name_list=save_name_list)