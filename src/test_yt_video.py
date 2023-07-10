import torch, timm
import torch.nn as nn
from models import CAVMAE, CAVMAEFT
assert timm.__version__ == '0.4.5'
import dataloader as dataloader
import argparse
from traintest_ft import train, validate
# debug
from torchsummary import summary
import pdb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ftmode", type=str, default='multimodal', help="how to fine-tune the model")
args = parser.parse_args()
args.loss_fn = nn.BCEWithLogitsLoss()

# Prepare audio and video dataset
data_val="/home/nano01/a/tao88/cav-mae/src/preprocess/sample_datafiles/sample_json_as.json"
label_csv="/home/nano01/a/tao88/cav-mae/src/preprocess/sample_datafiles/class_labels_indices_as.csv"
val_audio_conf = {'num_mel_bins': 128, 
                  'target_length': 1024, 
                  'freqm': 0, 'timem': 0, 'mixup': 0, 
                  'dataset': 'audioset',
                  'mode':'eval', 
                  'mean': -5.081, 'std': 4.4849, 
                  'noise': False, 'label_smooth': 0, 'im_res': 224}
val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(dataset_json_file=data_val, 
                               label_csv=label_csv, 
                               audio_conf=val_audio_conf),
    batch_size=32, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

# Load CAV-MAE model without decoder
model_path = "/home/nano01/a/tao88/cav-mae/pretrained_model/as-full-51.2.pth"
n_class = 527 # for audio set
audio_model = CAVMAEFT(label_dim=n_class,
                       modality_specific_depth=11)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(model_path, map_location=device)
audio_model = torch.nn.DataParallel(audio_model)
miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
audio_model.eval()
# print(miss, unexpected)

# Evaluate with pretrained model
total_frames = 10 # change if your total frame is different
multiframe_pred = []
for frame in range(total_frames):
    val_audio_conf['frame_use'] = frame
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(dataset_json_file=data_val, 
                                   label_csv=label_csv, 
                                   audio_conf=val_audio_conf),
        batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    stats, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
    audio_output, target = audio_output.numpy(), target.numpy()
    multiframe_pred.append(audio_output)

# multiframe_pred (10, 22, 547)
pdb.set_trace()


