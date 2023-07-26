We provide some of our training logs to help easy reproduction. You can compare your run with us to debug.

Pretrained Models:
- `vgg_pr_from_IN.txt` Pretrain on VGGSound from ImageNet initialization for 25 epochs. 
- `vgg_pr_from_as.txt` Do another round of SSL pretraining on VGGSound based on AS-2M pretrained model for 10 epochs with a smaller learning rate.

Finetuned Models

- `vgg_ft_(as)_pr_65.8.txt`: pretrained on AS-2M (scale++) and finetuned on vggsound, 65.8% accuracy on VGGSound.
- `vgg_ft_(as+vgg)_pr_65.9_log.txt`: pretrained on AS-2M (scale++), do another round pretraining on VGGSound (batch size 120), and finetune on VGGSound, 65.9% accuracy on VGGSound.
- `vgg_ft_(vgg)_pr_64.8.txt`: pretrained on VGGSound (from ImageNet initialization), and finetune on VGGSound, 64.8% accuracy on VGGSound.