# -*- coding: utf-8 -*-
# @Time    : 3/13/23 1:38 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : create_json_as.py

import os
import json

# "video_id": "-0nqfRcnAYE",
# "wav": "/data/sls/audioset/dave_version/audio/-0nqfRcnAYE.flac",
# "image": "/data/sls/audioset/dave_version/images/-0nqfRcnAYE.png",
# "labels": "/m/04brg2",
# "afeat": "/data/sls/scratch/yuangong/avbyol/data/audioset/a_feat/audio_feat_convnext_2/-0nqfRcnAYE.npy",
# "vfeat": "/data/sls/scratch/yuangong/avbyol/data/audioset/v_feat/video_feat_convnext_2/-0nqfRcnAYE.npy",
# "video": "/data/sls/audioset/dave_version/eval/-0nqfRcnAYE.mkv"

def clean_json(dataset_json_file):
    new_data = []
    with open(dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
        data = data_json['data']
        print('before clean {:d} files'.format(len(data)))
        for entry in data:
            wav = entry['wav']
            video_path = entry['image']
            video_path = video_path.split('/')[-1]
            print(video_path)
            video_id = video_path.split('/')[-1].split('.')[-2]
            video_path = '/data/sls/scratch/yuangong/avbyol2/egs/vggsound/preprocess/data/image_mulframe/'
            labels = entry['labels']

            new_entry = {}
            new_entry['video_id'] = video_id
            new_entry['wav'] = wav
            new_entry['video_path'] = video_path
            new_entry['labels'] = labels
            new_data.append(new_entry)

    output = {'data': new_data}
    print('after clean {:d} files'.format(len(new_data)))
    with open(dataset_json_file[:-5] + '_cleaned.json', 'w') as f:
        json.dump(output, f, indent=1)

clean_json('/data/sls/scratch/yuangong/cav-mae/pretrained_model/datafiles/vggsound/vgg_test_5_per_class_for_retrieval.json')