import torch, timm
from models import CAVMAE, CAVMAEFT
assert timm.__version__ == '0.4.5'
from torchsummary import summary
import pdb

# Load CAV-MAE pretrained model
model_path = "/home/nano01/a/tao88/cav-mae/pretrained_model/audio_model.25.pth"
# CAV-MAE model with decoder
audio_model = CAVMAE(audio_length=1024,
                     modality_specific_depth=11,
                     norm_pix_loss=True, 
                     tr_pos=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(model_path, map_location=device)
audio_model = torch.nn.DataParallel(audio_model) # it is important to convert the model to dataparallel object as all weights are saved in dataparallel format (i.e., in module.xxx)
miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
audio_model.eval()
print(miss, unexpected) # check if all weights are correctly loaded

# Prepare audio and video data
val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
pdb.set_trace()




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

pdb.set_trace()


