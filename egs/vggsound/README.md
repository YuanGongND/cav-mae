Pretraining Scripts
- `run_cavmae_pretrain.sh` Pretain on VGGSound from ImageNet initialization.
- `run_cavmae_pretrain_as.sh` Pretrain on VGGSound from AS-2M pretraining model (CAV-MAE Scale++).

Finetuning Script
- `run_cavmae_ft.sh` Finetune AS-2M pretrained CAV-MAE (Scale++) on VGGSound, should get ~65.8% accuracy.
