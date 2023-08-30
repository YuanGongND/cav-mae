# Contrastive Audio-Visual Masked Autoencoder
- [Introduction](#introduction)
- [Citing](#citing)
- [What's in this repo?](#whats-in-this-repo)
- [The CAV-MAE Model](#the-cav-mae-model)
- [Data Preparation](#data-preparation)
    - [Step 1. Extract audio track and image frames from the video](#step-1-extract-audio-track-and-image-frames-from-the-video)
    - [Step 2. Build a label set and json file for your dataset.](#step-2-build-a-label-set-and-json-file-for-your-dataset)
- [CAV-MAE Pretraining](#cav-mae-pretraining)
    - [Adapt vision-MAE checkpoint](#adapt-vision-mae-checkpoint)
    - [Build a virtual environment and install packages](#build-a-virtual-environment-and-install-packages)
    - [Run CAV-MAE pretraining](#run-cav-mae-pretraining)
    - [Do additional pretraining based on AS-2M pretrained CAV-MAE](#do-additional-pretraining-based-on-as-2m-pretrained-cav-mae)
- [Audio-Visual Event Classification](#audio-visual-event-classification)
  - [AudioSet](#audioset)
  - [VGGSound](#vggsound)
- [Retrieval](#retrieval)
- [Inpainting](#inpainting)
- [Pretrained Models](#pretrained-models)
    - [CAV-MAE Pretrained Models (Main)](#cav-mae-pretrained-models-main)
    - [CAV-MAE Pretrained Models (Ablation Study)](#cav-mae-pretrained-models-ablation-study)
    - [CAV-MAE Pretrained+Finetuned Models](#cav-mae-pretrainedfinetuned-models)
    - [AudioSet and VGGSound Data Lists](#audioset-and-vggsound-data-lists)
- [Contact](#contact)
 
## Introduction  

<p align="center"><img src="https://github.com/YuanGongND/cav-mae/blob/master/CAV-MAE_Poster.png?raw=true?raw=true" alt="Illustration of CAV-MAE." width="900"/></p>

**[[Paper]](https://openreview.net/pdf?id=QPtMRyk5rb)** **[[Review]](https://openreview.net/forum?id=QPtMRyk5rb)** **[[5-Minute Video]](https://recorder-v3.slideslive.com/?share=80147&s=29e0bafa-9193-4971-ac6f-e5a4cf7d69a0)** **[[Slides]](https://docs.google.com/presentation/d/1l-cofX9liikkVG6TkH2XwboYLVJoUDhpcyyq-fqtfcM/edit?usp=sharing)**
**[[MIT News](https://news.mit.edu/2023/scaling-audio-visual-learning-without-labels-0605)]**

This repository contains the official implementation (in PyTorch) of the **Contrastive Audio-Visual Masked Autoencoder (CAV-MAE)** proposed in the ICLR 2023 paper [Contrastive Audio-Visual Masked Autoencoder](https://openreview.net/forum?id=QPtMRyk5rb) (Yuan Gong, Andrew Rouditchenko, Alexander H. Liu, David Harwath, Leonid Karlinsky, Hilde Kuehne, James Glass).  

CAV-MAE **combines** two major self-supervised learning frameworks: **contrastive learning** and **masked data modeling**, to learn a joint and coordinated audio-visual representation. Our experiments show that the contrastive audio-visual correspondence learning objective not only enables the model to perform audio-visual retrieval tasks, but also helps the model learn a better joint representation. 
CAV-MAE achieves a new SOTA accuracy of 65.9% on VGGSound, and is comparable with the previous best supervised pretrained model on AudioSet in the audio-visual event classification task.

**Reviews:** The reviews of this paper and our responses are on [OpenReview](https://openreview.net/forum?id=QPtMRyk5rb), we thank the chair and anonymous reviewers' invaluable comments.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrastive-audio-visual-masked-autoencoder/multi-modal-classification-on-audioset)](https://paperswithcode.com/sota/multi-modal-classification-on-audioset?p=contrastive-audio-visual-masked-autoencoder)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrastive-audio-visual-masked-autoencoder/multi-modal-classification-on-vgg-sound)](https://paperswithcode.com/sota/multi-modal-classification-on-vgg-sound?p=contrastive-audio-visual-masked-autoencoder)

## Citing  
Please cite our paper if you find this repository useful.   
```  
@inproceedings{gong2023contrastive,
    title={Contrastive Audio-Visual Masked Autoencoder},
    author={Yuan Gong and Andrew Rouditchenko and Alexander H. Liu and David Harwath and Leonid Karlinsky and Hilde Kuehne and James R. Glass},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=QPtMRyk5rb}
}
```

## What's in this repo?

This repo contains everything you would need to reproduce our experiments and further adapt pretrained CAV-MAE to your task. Specifically, 

- The `CAVMAE` and `CAVMAEFT` model scripts are in [`src/models/cav-mae.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/models/cav_mae.py). 
- The data preprocessing scripts are in [`src/preprocess/`](https://github.com/YuanGongND/cav-mae/tree/master/src/preprocess). 
- The training pipelines are in [`src/run_cavmae_pretrain.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/run_cavmae_pretrain.py) and [`src/traintest_cavmae.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/traintest_cavmae.py) (for cav-mae self-supervised pretraining); and [`src/run_cavmae_ft.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/run_cavmae_ft.py) and [`src/traintest_ft.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/traintest_ft.py) (for classification finetuning).
- The AudioSet and VGGSound training scripts and logs are in [`egs/{audioset,vggsound}`](https://github.com/YuanGongND/cav-mae/tree/master/egs)
- The retrieval experiments script is in [`src/retrieval.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/retrieval.py).
- The inpainting experiments scripts is in [`src/inpainting.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/inpaint.py).
- The audio representation extraction scripts is in [`src/extract_audio_representation.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/extract_audio_representation.py).
- Pretrained models and data lists, a detailed list is [[here]](#pretrained-models).

## The CAV-MAE Model 

The proposed *cav-mae* models are in `src/models/cav-mae.py`, there are two models: `CAVMAE` (for pretraining, with decoder) and `CAVMAEFT` (for finetuning, without decoder). In general, the code is self-contained and commented, the best way to understand the details is by reading the code.

The input of `CAVMAE` and `CAVMAEFT` should be a pair of (audio, image). The shape of the audio should be [batch size, length_in_frame, mel_bin_num], e.g., [batch size, 1000, 128] ; the shape of the image should be [batch size, channel_num, height, width], e.g., [batch_size, 3, 224, 224].

The `CAVMAE` model, with its default arguments, is the model we used in the paper. However, there are some other things also implemented, e.g.,
- The masking ratios of audio and image do not have to be the same. 
- We used unstructured masking in the paper, but structured masking strategies (e.g., time, freq, time-freq) are also implemented, you can change it by changing `mask_mode` in the `forward` function.
- We use single-directional contrastive loss in the paper, but the bidirectional contrastive loss is also implemented.
- You can skip either contrastive loss or mae loss by changing `mae_loss_weight` and `contrast_loss_weight` in the `forward` function.

## Data Preparation

You can of course use the `CAV-MAE` model with your own training pipeline, but the input still needs to be identical to ours if you want to use our pretrained model. 

Otherwise, if you want to use our pipeline (including any of pretraining, classification, retrieval, and inpainting). You will need to prepare the data in the same format as us. 

#### Step 1. Extract audio track and image frames from the video

Suppose you have a set of videos (e.g., in `.mp4` format), you will need to first extract the audio track and image frames offline and save them on the disk, doing it on-the-fly usually dramatically increases the data loading overhead. In `src/preprocess/extract_{audio,video_frame}.py`, we include our code to do the extraction. Both scripts are simple, you will need to prepare a `csv` file containing a list of video paths (see `src/preprocess/sample_video_extract_list.csv` for an example) and `target_fold` (a single path) of your desired place to save the output. 

By default, we assume the `video_id` is the name of the video without extension and path, e.g., the `video_id` of `/path/test12345.mp4` is `test12345`. the output image frames will be saved at `target_fold/frame_{0-9}/video_id.jpg`, the output audio track will be saved at `target_fold/video_id.wav`.

The audio and image `target_fold` is better to be different, please record the `target_fold` for the next step. 

We provide a minimal example. The video and list are provided in this repo, you can just run to generate the frames and audio:
```python
cd cav-mae/src/preprocess
# extract video frames
python extract_video_frame.py -input_file_list sample_video_extract_list.csv -target_fold ./sample_frames
# extract audio tracks
python extract_audio.py  -input_file_list sample_video_extract_list.csv -target_fold ./sample_audio
```


#### Step 2. Build a label set and json file for your dataset.

You will need two files:

- A label csv file listing all labels. (see `src/preprocess/sample_datafiles/class_labels_indices_as.csv` as an example).
- A json file that have four keys for each sample (see `src/preprocess/sample_datafiles/sample_json_as.json` as an example):
  - `wav`: the absolute path to the audio track extracted in the previous step, e.g., `/data/sls/audioset/--4gqARaEJE.flac`
  - `video_id`: the video_id (i.e., the video filename without extension), e.g., `--4gqARaEJE` for video `--4gqARaEJE.mp4`.
  - `video_path`: the `target_fold` you used in the previous step, e.g., `/data/sls/audioset/`. Our pipeline will load from `video_path/frame_{0-9}/video_id.jpg`, not `video_path/video_id.jpg` So **make sure `video_path/frame_{0-9}/video_id.jpg` contains your image frames.**
  - `labels`: all labels of this sample, if more than one, use `,` to separate, must be consistent with the label csv file.
  - You can see how we automatically generate such json with `src/preprocess/create_json_as.py`.

To make this easier, we share our AudioSet and VGGSound datafiles at [here](#audioset-and-vggsound-data-lists), you can use/modify based on our files. The shared datafiles also show the exact sample ids we used for our experiments, which may be helpful for reproduction purposes.

## CAV-MAE Pretraining

#### Adapt vision-MAE checkpoint

As we mentioned in the paper, initializing CAV-MAE with ImageNet pretrained checkpoints leads to performance improvement. However, the original MAE checkpoint is for single modality while CAV-MAE is multi-modality. We use a script to adapt the original MAE checkpoint for CAV-MAE. You **do not** need to do it by yourself as our pipeline will take care of it. But `src/adapt_vmae_weights.py` is how we did that if you are interested. 

#### Build a virtual environment and install packages

Before running any experiments, please install the necessary packages for this project:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

#### Run CAV-MAE pretraining

The pretraining script is `egs/audioset/run_cavmae_pretrain_scale++.sh`, simply run it by `./run_cavmae_pretrain_scale++.sh`, and it will call `src/run_cavmae_pretrain.py`, which will call `src/traintest_cavmae.py`.

Note `egs/audioset/run_cavmae_pretrain_scale++.sh` should reproduce `CAV-MAE-Scale++` (not in the paper), it requires larger GPUs (4 X 48GB GPUs); For smaller GPUs (4 X {24,12}GB GPUs), we also include `egs/audioset/run_cavmae_pretrain_{scale+,base).sh`, which should reproduce `CAV-MAE-BASE`. 

You will need to change the following to your own datafiles prepared in the previous step:
- `tr_data`: training data json file.
- `te_data`: test data json file.
- `label-csv`: the label csv file.

Note: labels are still needed for self-supervised pretraining even though they are not used. This is because we use a single data loader. You can feed dummy labels for true-unlabeled data.

Some notable arguments in the pretraining scripts:
- `masking_ratio=0.75`: the masking rate for both audio and visual.
- `contrast_loss_weight=0.01`: weight for the contrastive loss, can be 0.
- `mae_loss_weight=1.0`: weight for the MAE loss, can be 0.
- `tr_pos=False`: use trainable positional embedding, should be `False`.
- `norm_pix_loss=True`: use pixel normalization for MAE. Set `False` for the inpainting model and `True` otherwise.
- `pretrain_path=None`: pretrain base on another pretrained checkpoint.

#### Do additional pretraining based on AS-2M pretrained CAV-MAE

One typical scenario is that you have a new dataset (e.g., VGGSound), you can of course finetune AudioSet pretrained CAV-MAE on your new dataset. But in this paper, we find it is better to first do another round of self-supervised pretraining on your dataset, and then do supervised fine-tuning. `egs/vggsound/run_cavmae_pretrain_as.sh` is a good example of how to do this.

## Audio-Visual Event Classification

### AudioSet 

The AudioSet classification scripts are in `egs/audioset/`

- `run_cavmae_ft_full.sh` finetune on full AudioSet-2M with both audio and visual data. Should reproduce 51.2 mAP.
- `run_cavmae_ft_bal.sh` finetune on balanced AudioSet-20K with both audio and visual data. Should reproduce 42.2 (in the paper: 42.0mAP).
- `run_cavmae_ft_bal_audioonly.sh` finetune on balanced AudioSet-20K with audio only. Should reproduce 38.3 mAP (in the paper: 37.7 mAP).

You will need to change the following to your own datafiles prepared in the previous step:
- `tr_data`: training data json file.
- `te_data`: test data json file.
- `label-csv`: the label csv file.

Some notable arguments are:

- `ftmode=multimodal`: use multimodal data or not, set `audioonly` to finetune an audio-only model.
- `pretrain_path`: the pretrained CAV-MAE checkpoint path.
- `freeze_base`: freeze the CAV-MAE and only train the newly initialized MLP layer, i.e., linear probing, should be False for e2e finetuning.
- `head_lr`: the ratio between newly initialized MLP layer params / pretrained CAV-MAE params. Always set > 1.
- `lrscheduler_start`, `lrscheduler_decay`, and `lrscheduler_step`: the learning rate scheduler, start from `lrscheduler_start` epoch, learning rate will decay by `lrscheduler_decay` every `lrscheduler_step` epochs.
- `wa`, `wa_start`, `wa_end`: model weight averaging params. If `wa=True`, weights of checkpoint `wa_start` to checkpoint `wa_end` will be averaged before evaluation. `wa_end` should be smaller than the number of total epochs.
- `dataset_mean` and `dataset_std`: audio spectrogram dataset-level mean and std, you can use our value for close datasets (e.g., we use the same values for AudioSet and VGGSound).
- `target_length`: input audio length in frame, i.e., 1000 for 10s audio.
- `freqm` and `timem`: the specaug parameters for audio. `freqm` should be around 48, and `timem` should be around 20% of `target_length` .

You should get slightly better results than the paper as by default we finetune based on the `CAV-MAE-Scale++` model pretrained with our newest GPUs. But if you wish, you can also reproduce the (lower) number in the paper as we also provide checkpoints of `CAV-MAE-Scale+` and `CAV-MAE-Base`. Note these models are of the same size, but trained with different batch sizes, in other words, the finetuning cost is the same, so unless for ablation study purposes, we suggest using our strongest pretrained CAV-MAE.

(Some) training logs are also provided at `egs/audioset/training_logs` to help reproduction.

### VGGSound

Very similar to AudioSet, the scripts are at `egs/vggsound/run_cavmae_ft.sh`. This should reproduce 65.8 accuracy (without addition VGGSound pretraining) and 65.9 accuracy (with additional VGGSound pretraining).

Training logs are also provided at `egs/vggsound/training_logs` to help reproduction.

## Retrieval 

Scripts for audio-visual retrieval are at `src/retrieval.py`, code is self-contained. You just need a CAV-MAE checkpoint and a dataset. 

## Inpainting

Scripts for audio-visual retrieval are at `src/inpaint.py`, code is self-contained. You just need a CAV-MAE checkpoint (trained without pixel normalization) and a dataset. 

## Pretrained Models

#### CAV-MAE Pretrained Models (Main)

We provide the following CAV-MAE models. Note by `scale++`, we mean the batch size, the model size is the same as `base` models, so the finetuning cost is the same for all models. Therefore, we recommend using CAV-MAE-Scale++ for all purposes except for inpainting. 

Load CAV-MAE models with a decoder using the following script:

```python3
import torch,timm
from models import CAVMAE
assert timm.__version__ == '0.4.5' # it is important to have right version of timm
model_path = 'the path to your model location'
# CAV-MAE model with decoder
audio_model = CAVMAE(audio_length=1024, \ # all models trained with 10s audio
                     modality_specific_depth=11, \ # all models trained with 11 modality-specific layers and 1 shared layer
                     norm_pix_loss=True, tr_pos=False) # most models are trained with pixel normalization and non-trainabe positional embedding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(model_path, map_location=device)
audio_model = torch.nn.DataParallel(audio_model) # it is important to convert the model to dataparallel object as all weights are saved in dataparallel format (i.e., in module.xxx)
miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
print(miss, unexpected) # check if all weights are correctly loaded
```

|      Model Name     | Batch Size | Lambda_c |   Masking Ratio  |                     Usage                     |
|:-------------------:|:----------:|:--------:|:----------------:|:---------------------------------------------:|
|   **[CAV-MAE-Scale++ (Recommended)](https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1)** <br />[[China Mirror Link]](https://share.weiyun.com/vGfP1YZt) |     256    |   0.01   | 75% Unstructured | Recommend for all purposes except inpainting. |
|    [CAV-MAE-Scale+](https://www.dropbox.com/s/xu8bfie6hz86oev/audio_model.25.pth?dl=1)   |     108    |   0.01   | 75% Unstructured |            Reproduce the exact paper results.     |
|     [CAV-MAE-Base](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1)    |     48     |   0.01   | 75% Unstructured |            Reproduce the exact paper results.     |
| [CAV-MAE-Base-NoNorm](https://www.dropbox.com/s/arrg7cb3e4hhjwu/cav-mae-base-nonorm.pth?dl=1) |     48     |   0.01   | 75% Unstructured |                   Inpainting                  |

#### CAV-MAE Pretrained Models (Ablation Study)

In addition to the above models, we also release the following models for ablation studies. In each table, the models are trained with same setting except for the hyper-parameter of interest.

*Masking Ratio*

|    Model Name   | Batch Size | Lambda_c |   Masking Ratio  |
|:---------------:|:----------:|:--------:|:----------------:|
| [CAV-MAE-Base-50](https://www.dropbox.com/s/dgorer0ybdbnvgf/50.pth?dl=1) |     48     |   0.01   | 50% Unstructured |
| [CAV-MAE-Base-65](https://www.dropbox.com/s/xmuyksqch6l6g87/65.pth?dl=1) |     48     |   0.01   | 65% Unstructured |
|   [CAV-MAE-Base](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1)  |     48     |   0.01   | 75% Unstructured |
| [CAV-MAE-Base-85](https://www.dropbox.com/s/y3o9zggpecmrbnu/85.pth?dl=1) |     48     |   0.01   | 85% Unstructured |

*Audio Masking Method*

|         Model Name        | Batch Size | Lambda_c |        Masking Ratio       |
|:-------------------------:|:----------:|:--------:|:--------------------------:|
| [CAV-MAE-Base-Unstructured](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1) |     48     |   0.01   |      75% Unstructured      |
|     [CAV-MAE-Base-TF](https://www.dropbox.com/s/madv6ynuy5113zh/time_freq_75.pth?dl=1)      |     48     |   0.01   | 75% Time-Frequency Masking |
|     [CAV-MAE-Base-TF-50](https://www.dropbox.com/s/nd7qlagn8je6zjn/time_freq_50.pth?dl=1)    |     48     |   0.01   | 50% Time-Frequency Masking |
|       [CAV-MAE-Base-T](https://www.dropbox.com/s/hfehd7m379ehr0y/time_75.pth?dl=1)      |     48     |   0.01   |      75% Time Masking      |
|       [CAV-MAE-Base-F](https://www.dropbox.com/s/ad4fhzt6d3xre5p/freq_75.pth?dl=1)      |     48     |   0.01   |    75% Frequency Masking   |

*Contrastive Loss Weight*

|      Model Name     | Batch Size | Lambda_c |   Masking Ratio  |
|:-------------------:|:----------:|:--------:|:----------------:|
| [CAV-MAE-Base-C0.001](https://www.dropbox.com/s/4j4qyiyjcmbtc6u/cav-mae-base-c0.001.pth?dl=1) |     48     |   0.001  | 75% Unstructured |
|     [CAV-MAE-Base](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1)    |     48     |   0.01   | 75% Unstructured |
|  [CAV-MAE-Base-C0.1](https://www.dropbox.com/s/wf54ver9rs9fl1b/cav-mae-base-c0.1.pth?dl=1)  |     48     |   0.1   | 75% Unstructured |

*Symmetric (Bi-Directional) Contrastive Loss*

|      Model Name     | Batch Size | Lambda_c | Contrastive Loss |
|:-------------------:|:----------:|:--------:|:----------------:|
|    [CAV-MAE-Scale+](https://www.dropbox.com/s/mvhmg7eda410phr/single_direction.pth?dl=1)   |     120    |   0.01   | Single Direction |
| [CAV-MAE-Symc-Scale+](https://www.dropbox.com/s/ute6ydkw4hdv7rn/symc.pth?dl=1) |     120    |   0.01   |  Bi-Directional  |

#### CAV-MAE Pretrained+Finetuned Models

Load CAV-MAE models without a decoder (typically a finetuned model) using the following script:

```python3
import torch,timm
from models import CAVMAEFT
assert timm.__version__ == '0.4.5' # it is important to have right version of timm
model_path = 'the path to your model location'
n_class = 527 # 527 for audioset finetuned models, 309 for vggsound finetuned models
# CAV-MAE model without decoder
audio_model = models.CAVMAEFT(label_dim=n_class, \
                              modality_specific_depth=11)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(model_path, map_location=device)
audio_model = torch.nn.DataParallel(audio_model) # it is important to convert the model to dataparallel object as all weights are saved in dataparallel format (i.e., in module.xxx)
miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
print(miss, unexpected) # check if all weights are correctly loaded, if you are loading a model with decoder, you will see decoders are unexpected and missed are newly initialized classification heads
```

*Multi-Modal Audio-Visual Models*

| Pretrained Model |  Pretrain Data | Finetune Data |  Performance  | Download |
|:----------------:|:----------------:|:--------------:|:-------------:|:--------:|
|  CAV-MAE-Scale+  |    AudioSet-2M   |   AudioSet-2M  |    51.2 mAP   |  [link](https://www.dropbox.com/s/hejj49zdgyyfsuh/as-full-51.2.pth?dl=1)  |
|  CAV-MAE-Scale+  |    AudioSet-2M   |  AudioSet-20K  |    42.0 mAP   |  [link](https://www.dropbox.com/s/9x6y642so3zve5y/as-20k-42.0.pth?dl=1)  |
|  CAV-MAE-Scale+  |    AudioSet-2M   |    VGGSound    | 65.5 accuracy |  [link](https://www.dropbox.com/s/f4wrbxv2unewss9/vgg_65.5.pth?dl=1)  |
|  CAV-MAE-Scale+  | AS-2M + VGGSound |    VGGSound    | 65.9 accuracy |  [link](https://www.dropbox.com/s/q66s3mu526mj2x6/vgg_66.0.pth?dl=1)  |

*Audio-Only or Visual-Only Models*

One conclusion of the paper is multi-modal pretraining helps single-modality performance. Below we release models pretrained with multi-modal data, but fine-tuned on only audio or only visual data. These models works when **only one** modality is input in inference.

| Pretrained Model |      Pretrain Data      |    Finetune Data    |  Performance  | Download |
|:----------------:|:-------------------------:|:--------------------:|:-------------:|:--------:|
|  CAV-MAE-Scale+  | AudioSet-2M (multi-modal) |  AudioSet-2M (audio) |    46.6 mAP   |  [link](https://www.dropbox.com/s/itfw7p0ueq7z9og/as_46.6.pth?dl=1)  |
|  CAV-MAE-Scale+  | AudioSet-2M (multi-modal) | AudioSet-20K (audio) |    37.7 mAP   |  [link](https://www.dropbox.com/s/pariabyh1iyayda/as_37.7.pth?dl=1)  |
|  CAV-MAE-Scale++ <br> (Note it is ++)  | AudioSet-2M (multi-modal) | AudioSet-20K (visual) |   20.0 mAP   |  [link](https://www.dropbox.com/s/9ngkq9ygwqecxz5/as_20.0.pth?dl=1)  |
|  CAV-MAE-Scale+  | AS-2M + VGGSound (multi-modal) |   VGGSound (audio) | 59.8 accuracy |  [link](https://www.dropbox.com/s/l4rj0sgpnt08bp2/vgg_59.8.pth?dl=1)  |

#### AudioSet and VGGSound Data Lists

We also release the following AudioSet and VGGSound data lists. The datafiles can be used for 1) use these as samples for your data preparation; and 2) check the exact sample ids we used in the paper for reproduction.
Due to copyright reasons, we cannot provide raw data for both datasets.

*AudioSet*

|          Filename         |                              Content                             |
|:-------------------------:|:----------------------------------------------------------------:|
|   [AudioSet Label CSV File](https://www.dropbox.com/s/z3ko8bv9b7738t1/class_labels_indices.csv?dl=1)      |                     The label set of AudioSet                    |
|   [AudioSet-2M Json File](https://www.dropbox.com/s/18hoeq92juqsg2g/audioset_2m_cleaned.json?dl=1)   |   Full AudioSet training set sample info we used in this paper   |
|   [AudioSet-2M Sample Weight List](https://www.dropbox.com/s/y3omtsq4qjeujov/audioset_2m_cleaned_weight.csv?dl=1)   |   Class-balancing weight for each sample, in the same order with `AudioSet-2M Json File`   |
|   [AudioSet-20K Json File](https://www.dropbox.com/s/6mds4ld24kk42o3/audioset_20k_cleaned.json?dl=1)  | Balanced AudioSet training set sample info we used in this paper |
|   [AudioSet-Eval Json File](https://www.dropbox.com/s/4e39racy7ys1c5o/audioset_eval_cleaned.json?dl=1)  |        AudioSet eval set sample info we used in this paper       |
|   [AudioSet Retrieval Subset](https://www.dropbox.com/s/bb9p6t44t6etrzc/audioset_eval_5_per_class_for_retrieval_cleaned.json?dl=1) |          AudioSet retrieval subset (5 samples per class) we used in this paper         |

*VGGSound*

|          Filename         |                               Content                               |
|:-------------------------:|:-------------------------------------------------------------------:|
|  [VGGSound Label CSV File](https://www.dropbox.com/s/6k0czs8pzz6yj2c/class_labels_indices_vgg.csv?dl=1)  |                      The label set of VGGSound                      |
|  [VGGSound Train Json File](https://www.dropbox.com/s/0khhenoh35lkvym/vgg_train_cleaned.json?dl=1) |       VGGSound train set sample info we used in this paper      |
|  [VGGSound Train Sample Weight List](https://www.dropbox.com/s/8ynyvpyqm5wtluq/vgg_train_cleaned_weight.csv?dl=1) |       Class-balancing weight for each sample, in the same order with `VGGSound Train Json File`     |
|  [VGGSound Test Json File](https://www.dropbox.com/s/g9zfrrzx57t5hlx/vgg_test_cleaned.json?dl=1)  |         VGGSound test set sample info we used in this paper         |
| [VGGSound Retrieval Subset](https://www.dropbox.com/s/exrdfh2nfrt1qvo/vgg_test_5_per_class_for_retrieval_cleaned.json?dl=1) | VGGSound retrieval subset(5 sample per class) we used in this paper |

 ## Contact
If you have a question, please bring up an issue (preferred) or send me an email [yuangong@mit.edu](yuangong@mit.edu).