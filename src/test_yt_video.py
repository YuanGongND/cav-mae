import torch, timm
import torch.nn as nn
from models import CAVMAE, CAVMAEFT
assert timm.__version__ == '0.4.5'
import dataloader as dataloader
import argparse
from traintest_ft import train, validate
import cv2
import numpy as np
import pandas as pd
from torchsummary import summary
import pdb # debug

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
# multiframe_pred = []
# multiframe_target = []
for frame in range(total_frames):
    val_audio_conf['frame_use'] = frame
    val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(dataset_json_file=data_val, 
                                                                        label_csv=label_csv, 
                                                                        audio_conf=val_audio_conf),
                                             batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    stats, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
    audio_output, target = audio_output.numpy(), target.numpy()
    if frame == 0:
        multiframe_pred = np.expand_dims(audio_output, axis=0) # (1, 22, 547)
    else:
        multiframe_pred = np.concatenate((multiframe_pred, np.expand_dims(audio_output, axis=0)), axis=0)


# multiframe_pred (10, 22, 547), 10 frames, 22 10-sec video clips, 547 classes
# 307 car, 137 music, 0 speech, 443 shatter, 500 silence, 72 animal, 300 vehicle

multiframe_pred_class = np.argmax(multiframe_pred, axis=2) # (10, 22)
multiframe_pred_class_top3 = np.argsort(multiframe_pred, axis=2)[:,:,::-1][:,:,:3] 
idx_to_classname = pd.read_csv(label_csv, usecols=['index', 'mid', 'display_name'])
# Put text annotation to the video frames and watch the video
vid_cap = cv2.VideoCapture('/home/nano01/a/tao88/cav-mae/src/preprocess/sample_video/carsdontfly.mp4')
frame_count = 0
# img_array = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    vid_cap.set(cv2.CAP_PROP_POS_MSEC, frame_count*1000) # interval = 1000ms, 1 fps
    ret, frame = vid_cap.read()
    if not ret: break
    try:
        class_idx = multiframe_pred_class_top3[frame_count%10, frame_count//10]
        class_text_1 = idx_to_classname.iloc[class_idx[0], 2]
        class_text_2 = idx_to_classname.iloc[class_idx[1], 2]
        class_text_3 = idx_to_classname.iloc[class_idx[2], 2]          
    except:
        class_text_1, class_text_1, class_text_3 = 'Not inferred', 'Not inferred', 'Not inferred'
    cv2.putText(frame, class_text_1, (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_4)
    cv2.putText(frame, class_text_2, (50, 90), font, 1, (0, 255, 255), 3, cv2.LINE_4)
    cv2.putText(frame, class_text_3, (50, 130), font, 1, (0, 255, 255), 3, cv2.LINE_4)
    # img_array.append(frame)
    cv2.imwrite('/home/nano01/a/tao88/cav-mae/src/preprocess/processed_frames/frame_{}.png'.format(frame_count),
                frame)
    # cv2.imshow('Video-audio event classification', frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the cap object
vid_cap.release()
# close all windows
cv2.destroyAllWindows()

# size = (1280, 720)
# vid_out = cv2.VideoWriter(filename='/home/nano01/a/tao88/cav-mae/src/preprocess/sample_video/carsdontfly_processed.mp4',
#                       fourcc=cv2.VideoWriter_fourcc(*'avc1'),
#                       fps=1,
#                       frameSize=size,
#                       isColor=True)
# for i in range(len(img_array)):
#     vid_out.write(img_array[i])
# vid_out.release()
# cv2.destroyAllWindows()
# print('\nAll {} frames captured and processed. Video generated and saved.'.format(frame_count))

pdb.set_trace()


