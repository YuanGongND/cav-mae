# CAV-MAE Pretrained Model Zoo
## CAV-MAE Pretrained Models (Main)

We provide the following CAV-MAE models. Note by `scale++`, we mean the batch size, the model size is the same with `base` models, so the finetuning cost is same for all models. Therefore, we recommend to use CAV-MAE-Scale++ for all purposes except for inpainint. 

Load CAV-MAE models with a decoder using the following script:

```python3
import torch
from models import CAVMAE
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
|    [CAV-MAE-Scale+](https://www.dropbox.com/s/xu8bfie6hz86oev/audio_model.25.pth?dl=1)   |     108    |   0.01   | 75% Unstructured |            Reproduce exact paper results.     |
|     [CAV-MAE-Base](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1)    |     48     |   0.01   | 75% Unstructured |            Reproduce exact paper results.     |
| [CAV-MAE-Base-NoNorm](https://www.dropbox.com/s/arrg7cb3e4hhjwu/cav-mae-base-nonorm.pth?dl=1) |     48     |   0.01   | 75% Unstructured |                   Inpainting                  |

## CAV-MAE Pretrained Models (Ablation Study)

In addition to the above models, we also release the following models for ablation studies. In each table, the models are trained with same setting except for the hyper-parameter of interest.

### Masking Ratio

|    Model Name   | Batch Size | Lambda_c |   Masking Ratio  |
|:---------------:|:----------:|:--------:|:----------------:|
| [CAV-MAE-Base-50](https://www.dropbox.com/s/dgorer0ybdbnvgf/50.pth?dl=1) |     48     |   0.01   | 50% Unstructured |
| [CAV-MAE-Base-65](https://www.dropbox.com/s/xmuyksqch6l6g87/65.pth?dl=1) |     48     |   0.01   | 65% Unstructured |
|   [CAV-MAE-Base](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1)  |     48     |   0.01   | 75% Unstructured |
| [CAV-MAE-Base-85](https://www.dropbox.com/s/y3o9zggpecmrbnu/85.pth?dl=1) |     48     |   0.01   | 85% Unstructured |

### Audio Masking Method

|         Model Name        | Batch Size | Lambda_c |        Masking Ratio       |
|:-------------------------:|:----------:|:--------:|:--------------------------:|
| [CAV-MAE-Base-Unstructured](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1) |     48     |   0.01   |      75% Unstructured      |
|     [CAV-MAE-Base-TF](https://www.dropbox.com/s/madv6ynuy5113zh/time_freq_75.pth?dl=1)      |     48     |   0.01   | 75% Time-Frequency Masking |
|     [CAV-MAE-Base-TF-50](https://www.dropbox.com/s/nd7qlagn8je6zjn/time_freq_50.pth?dl=1)    |     48     |   0.01   | 50% Time-Frequency Masking |
|       [CAV-MAE-Base-T](https://www.dropbox.com/s/hfehd7m379ehr0y/time_75.pth?dl=1)      |     48     |   0.01   |      75% Time Masking      |
|       [CAV-MAE-Base-F](https://www.dropbox.com/s/ad4fhzt6d3xre5p/freq_75.pth?dl=1)      |     48     |   0.01   |    75% Frequency Masking   |

### Contrastive Loss Weight

|      Model Name     | Batch Size | Lambda_c |   Masking Ratio  |
|:-------------------:|:----------:|:--------:|:----------------:|
| [CAV-MAE-Base-C0.001](https://www.dropbox.com/s/4j4qyiyjcmbtc6u/cav-mae-base-c0.001.pth?dl=1) |     48     |   0.001  | 75% Unstructured |
|     [CAV-MAE-Base](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1)    |     48     |   0.01   | 75% Unstructured |
|  [CAV-MAE-Base-C0.1](https://www.dropbox.com/s/wf54ver9rs9fl1b/cav-mae-base-c0.1.pth?dl=1)  |     48     |   0.1   | 75% Unstructured |

### Symmetric (Bi-Directional) Contrastive Loss

|      Model Name     | Batch Size | Lambda_c | Contrastive Loss |
|:-------------------:|:----------:|:--------:|:----------------:|
|    [CAV-MAE-Scale+](https://www.dropbox.com/s/mvhmg7eda410phr/single_direction.pth?dl=1)   |     120    |   0.01   | Single Direction |
| [CAV-MAE-Symc-Scale+](https://www.dropbox.com/s/ute6ydkw4hdv7rn/symc.pth?dl=1) |     120    |   0.01   |  Bi-Directional  |


## CAV-MAE Pretrained+Finetuned Models

```python3
import torch
from models import CAVMAEFT
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

### Multi-Modal Audio-Visual Models

| Pretrained Model |  Pretraine Data | Finetune Data |  Performance  | Download |
|:----------------:|:----------------:|:--------------:|:-------------:|:--------:|
|  CAV-MAE-Scale+  |    AudioSet-2M   |   AudioSet-2M  |    51.2 mAP   |  [link](https://www.dropbox.com/s/hejj49zdgyyfsuh/as-full-51.2.pth?dl=1)  |
|  CAV-MAE-Scale+  |    AudioSet-2M   |  AudioSet-20K  |    42.0 mAP   |  [link](https://www.dropbox.com/s/9x6y642so3zve5y/as-20k-42.0.pth?dl=1)  |
|  CAV-MAE-Scale+  |    AudioSet-2M   |    VGGSound    | 65.5 accuracy |  [link](https://www.dropbox.com/s/f4wrbxv2unewss9/vgg_65.5.pth?dl=1)  |
|  CAV-MAE-Scale+  | AS-2M + VGGSound |    VGGSound    | 65.9 accuracy |  [link](https://www.dropbox.com/s/q66s3mu526mj2x6/vgg_66.0.pth?dl=1)  |

### Audio-Only or Visual-Only Models

One conclusion of the paper is multi-modal pretraining helps single-modality performance. Below we release models pretrained with multi-modal data, but fine-tuned on only audio or only visual data. These models works when **only one** modality is input in inference.

| Pretrained Model |      Pretraine Data      |    Finetune Data    |  Performance  | Download |
|:----------------:|:-------------------------:|:--------------------:|:-------------:|:--------:|
|  CAV-MAE-Scale+  | AudioSet-2M (multi-modal) |  AudioSet-2M (audio) |    46.6 mAP   |  [link](https://www.dropbox.com/s/itfw7p0ueq7z9og/as_46.6.pth?dl=1)  |
|  CAV-MAE-Scale+  | AudioSet-2M (multi-modal) | AudioSet-20K (audio) |    37.7 mAP   |  [link](https://www.dropbox.com/s/pariabyh1iyayda/as_37.7.pth?dl=1)  |
|  CAV-MAE-Scale++ <br> (Note it is ++) | AudioSet-2M (multi-modal) | AudioSet-20K (visual) |   20.0 mAP   |  [link](https://www.dropbox.com/s/9ngkq9ygwqecxz5/as_20.0.pth?dl=1)  |
|  CAV-MAE-Scale+  | AS-2M + VGGSound (multi-modal) |   VGGSound (audio)   | 59.8 accuracy |  [link](https://www.dropbox.com/s/l4rj0sgpnt08bp2/vgg_59.8.pth?dl=1)  |

## AudioSet and VGGSound Data Lists

We also release the following AudioSet and VGGSound data lists. The datafiles can be used for 1) use these as samples for your data preparation; and 2) check the exact sample ids we used in the paper for reproduction.
Due to copyright reasons, we cannot provide raw data for both datasets.

### AudioSet

|          Filename         |                              Content                             |
|:-------------------------:|:----------------------------------------------------------------:|
|   [AudioSet Label CSV File](https://www.dropbox.com/s/z3ko8bv9b7738t1/class_labels_indices.csv?dl=1)      |                     The label set of AudioSet                    |
|   [AudioSet-2M Json File](https://www.dropbox.com/s/18hoeq92juqsg2g/audioset_2m_cleaned.json?dl=1)   |   Full AudioSet training set sample info we used in this paper   |
|   [AudioSet-2M Sample Weight List](https://www.dropbox.com/s/y3omtsq4qjeujov/audioset_2m_cleaned_weight.csv?dl=1)   |   Class-balancing weight for each sample, in the same order with `AudioSet-2M Json File`   |
|   [AudioSet-20K Json File](https://www.dropbox.com/s/6mds4ld24kk42o3/audioset_20k_cleaned.json?dl=1)  | Balanced AudioSet training set sample info we used in this paper |
|   [AudioSet-Eval Json File](https://www.dropbox.com/s/4e39racy7ys1c5o/audioset_eval_cleaned.json?dl=1)  |        AudioSet eval set sample info we used in this paper       |
|   [AudioSet Retrieval Subset](https://www.dropbox.com/s/bb9p6t44t6etrzc/audioset_eval_5_per_class_for_retrieval_cleaned.json?dl=1) |          AudioSet retrieval subset (5 sample per class) we used in this paper         |

### VGGSound

|          Filename         |                               Content                               |
|:-------------------------:|:-------------------------------------------------------------------:|
|  [VGGSound Label CSV File](https://www.dropbox.com/s/6k0czs8pzz6yj2c/class_labels_indices_vgg.csv?dl=1)  |                      The label set of VGGSound                      |
|  [VGGSound Train Json File](https://www.dropbox.com/s/0khhenoh35lkvym/vgg_train_cleaned.json?dl=1) |       VGGSound train set set sample info we used in this paper      |
|  [VGGSound Train Sample Weight List](https://www.dropbox.com/s/8ynyvpyqm5wtluq/vgg_train_cleaned_weight.csv?dl=1) |       Class-balancing weight for each sample, in the same order with `VGGSound Train Json File`     |
|  [VGGSound Test Json File](https://www.dropbox.com/s/g9zfrrzx57t5hlx/vgg_test_cleaned.json?dl=1)  |         VGGSound test set sample info we used in this paper         |
| [VGGSound Retrieval Subset](https://www.dropbox.com/s/exrdfh2nfrt1qvo/vgg_test_5_per_class_for_retrieval_cleaned.json?dl=1) | VGGSound retrieval subset(5 sample per class) we used in this paper |
