Pretrain Logs:
- `as2m_pretrain_base.txt` pretrain a CAV-MAE model from ImageNet initialization and with a batch size of 48 (12 on each GPU, a total of 4 GPUs, i.e., the `base` setting).

Finetuning Logs:
- `as20k_ft_42.2.txt` finetune CAV-MAE Scale++ on AS-20K, 42.2 mAP on AS-Eval.
- `as20k_ft_audioonly_38.2.txt` finetune CAV-MAE Scale++ on AS-20K with only audio data, 38.2 mAP on AS-Eval with only audio data.
- `as20k_ft_visualonly_20.0.txt` finetune CAV-MAE Scale++ on AS-20K with only visual data (10 frames per 10-second video), 20,9 mAP on AS-Eval with only audio data.
- `as2m_ft_51.1.txt` finetune CAV-MAE Scale++ on AS-2M, 51.1 mAP on AS-Eval.
- `as2m_ft_audioonly_46.6.txt` finetune CAV-MAE Scale++ on AS-2M with only audio data, 46.6 mAP on AS-Eval with only audio data.