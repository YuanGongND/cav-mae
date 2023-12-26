# -*- coding: utf-8 -*-
# @Time    : 3/11/23 4:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cav_mae.py

import os

os.environ["TORCH_HOME"] = "./pretrained_models"
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == "a":
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == "v":
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x


# our main proposed model, for pretraining only, for finetuning, use CAVMAEFT class
class CAVMAE(nn.Module):
    """CAV-MAE Model"""

    def __init__(
        self,
        img_size=224,
        audio_length=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        modality_specific_depth=11,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        tr_pos=False,
    ):
        super().__init__()
        print("A CAV-MAE Model")
        print("Use norm_pix_loss: ", norm_pix_loss)
        print("Learnable Positional Embedding: ", tr_pos)

        # the encoder part
        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print(
            "Number of Audio Patches: {:d}, Visual Patches: {:d}".format(
                self.patch_embed_a.num_patches, self.patch_embed_v.num_patches
            )
        )

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(
            torch.zeros(1, self.patch_embed_a.num_patches, embed_dim),
            requires_grad=tr_pos,
        )  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(
            torch.zeros(1, self.patch_embed_v.num_patches, embed_dim),
            requires_grad=tr_pos,
        )  # fixed sin-cos embedding

        # audio-branch
        self.blocks_a = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(modality_specific_depth)
            ]
        )
        # visual-branch
        self.blocks_v = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(modality_specific_depth)
            ]
        )
        # unified branch
        self.blocks_u = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(12 - modality_specific_depth)
            ]
        )

        # independent normalization layer for audio, visual, and audio-visual
        self.norm_a, self.norm_v, self.norm = (
            norm_layer(embed_dim),
            norm_layer(embed_dim),
            norm_layer(embed_dim),
        )

        # the decoder part
        # Project to lower dimension for the decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # token used for masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_modality_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_a = nn.Parameter(
            torch.zeros(1, self.patch_embed_a.num_patches, decoder_embed_dim),
            requires_grad=tr_pos,
        )  # fixed sin-cos embedding
        self.decoder_pos_embed_v = nn.Parameter(
            torch.zeros(1, self.patch_embed_v.num_patches, decoder_embed_dim),
            requires_grad=tr_pos,
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # project channel is different for two modality, use two projection head
        self.decoder_pred_a = nn.Linear(
            decoder_embed_dim, patch_size**2 * 1, bias=True
        )  # decoder to patch
        self.decoder_pred_v = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("Audio Positional Embedding Shape:", self.pos_embed_a.shape)
        print("Visual Positional Embedding Shape:", self.pos_embed_v.shape)

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        pos_embed_a = get_2d_sincos_pos_embed(
            self.pos_embed_a.shape[-1],
            8,
            int(self.patch_embed_a.num_patches / 8),
            cls_token=False,
        )
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(
            self.pos_embed_v.shape[-1],
            int(self.patch_embed_v.num_patches**0.5),
            int(self.patch_embed_v.num_patches**0.5),
            cls_token=False,
        )
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        decoder_pos_embed_a = get_2d_sincos_pos_embed(
            self.decoder_pos_embed_a.shape[-1],
            8,
            int(self.patch_embed_a.num_patches / 8),
            cls_token=False,
        )
        self.decoder_pos_embed_a.data.copy_(
            torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0)
        )

        decoder_pos_embed_v = get_2d_sincos_pos_embed(
            self.decoder_pos_embed_v.shape[-1],
            int(self.patch_embed_v.num_patches**0.5),
            int(self.patch_embed_v.num_patches**0.5),
            cls_token=False,
        )
        self.decoder_pos_embed_v.data.copy_(
            torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.modality_a, std=0.02)
        torch.nn.init.normal_(self.modality_v, std=0.02)
        torch.nn.init.normal_(self.decoder_modality_a, std=0.02)
        torch.nn.init.normal_(self.decoder_modality_v, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, c, h, w, p=16):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x, c, h, w, p=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking_unstructured(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_structured(self, x, mask_ratio, t=64, f=8, mode="time"):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        assert L == f * t
        noise = noise.reshape(N, f, t)  # the audio patch is in shape [f,t], not [t,f]
        if mode == "time":
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
        elif mode == "freq":
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        elif mode == "tf":
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio * 0.7))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio * 0.7))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        noise = noise.reshape(N, L)

        # sort noise for each sample, only need to manuplate these two ids_shuffle, ids_restore
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, a, v, mask_ratio_a, mask_ratio_v, mask_mode="unstructured"
    ):
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # by default, we always use unstructured masking
        if mask_mode == "unstructured":
            a, mask_a, ids_restore_a = self.random_masking_unstructured(a, mask_ratio_a)
        # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
        else:
            a, mask_a, ids_restore_a = self.random_masking_structured(
                a, mask_ratio_a, t=64, f=8, mode=mask_mode
            )

        # visual branch always use unstructured masking
        v, mask_v, ids_restore_v = self.random_masking_unstructured(v, mask_ratio_v)

        # audio and visual stream, independent blocks
        for blk in self.blocks_a:
            a = blk(a)

        for blk in self.blocks_v:
            v = blk(v)

        x = torch.cat((a, v), dim=1)

        # unified stream, shared blocks_u, but independent normalization layers
        for blk in self.blocks_u:
            x = blk(x)
        x = self.norm(x)

        for blk in self.blocks_u:
            ca = blk(a, "a")
        ca = self.norm_a(ca)

        for blk in self.blocks_u:
            cv = blk(v, "v")
        cv = self.norm_v(cv)

        return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv

    def forward_decoder(self, x, mask_a, ids_restore_a, mask_v, ids_restore_v):
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
        mask_tokens_a = self.mask_token.repeat(x.shape[0], int(mask_a[0].sum()), 1)
        a_ = torch.cat(
            [
                x[:, : self.patch_embed_a.num_patches - int(mask_a[0].sum()), :],
                mask_tokens_a,
            ],
            dim=1,
        )  # no cls token
        a_ = torch.gather(
            a_, dim=1, index=ids_restore_a.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        # similar for the visual modality
        mask_tokens_v = self.mask_token.repeat(x.shape[0], int(mask_v[0].sum()), 1)
        v_ = torch.cat(
            [
                x[:, self.patch_embed_a.num_patches - int(mask_a[0].sum()) :, :],
                mask_tokens_v,
            ],
            dim=1,
        )  # no cls token
        v_ = torch.gather(
            v_, dim=1, index=ids_restore_v.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        # concatenate audio and visual tokens
        x = torch.cat([a_, v_], dim=1)

        decoder_pos_embed = torch.cat(
            [self.decoder_pos_embed_a, self.decoder_pos_embed_v], dim=1
        )
        x = x + decoder_pos_embed

        # add modality indication tokens
        x[:, 0 : self.patch_embed_a.num_patches, :] = (
            x[:, 0 : self.patch_embed_a.num_patches, :] + self.decoder_modality_a
        )
        x[:, self.patch_embed_a.num_patches :, :] = (
            x[:, self.patch_embed_a.num_patches :, :] + self.decoder_modality_v
        )

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x_a = self.decoder_pred_a(x[:, : self.patch_embed_a.num_patches, :])
        x_v = self.decoder_pred_v(x[:, self.patch_embed_a.num_patches :, :])

        # return audio and video tokens
        return x_a, x_v

    def forward_contrastive(self, audio_rep, video_rep, bidirect_contrast=False):
        # calculate nce loss for mean-visual representation and mean-audio representation

        audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
        video_rep = torch.nn.functional.normalize(video_rep, dim=-1)

        total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05

        # by default we use single directional
        if bidirect_contrast == False:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            c_acc = (
                torch.sum(
                    torch.eq(
                        torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0),
                        torch.arange(0, total.shape[0], device=audio_rep.device),
                    )
                )
                / total.shape[0]
            )
            return nce, c_acc
        else:
            nce_1 = -torch.mean(
                torch.diag(torch.nn.functional.log_softmax(total, dim=0))
            )
            nce_2 = -torch.mean(
                torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0))
            )
            c_acc_1 = (
                torch.sum(
                    torch.eq(
                        torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0),
                        torch.arange(0, total.shape[0], device=audio_rep.device),
                    )
                )
                / total.shape[0]
            )
            c_acc_2 = (
                torch.sum(
                    torch.eq(
                        torch.argmax(
                            torch.nn.functional.softmax(total.t(), dim=0), dim=0
                        ),
                        torch.arange(0, total.shape[0], device=audio_rep.device),
                    )
                )
                / total.shape[0]
            )
            nce = (nce_1 + nce_2) / 2
            c_acc = (c_acc_1 + c_acc_2) / 2
            return nce, c_acc

    def forward_mae_loss(self, input, pred, mask, modality):
        if modality == "a":
            # for audio, need to adjust the shape
            input = input.unsqueeze(1)
            input = input.transpose(2, 3)
            target = self.patchify(
                input,
                1,
                int(input.shape[2] / self.patch_embed_a.patch_size[0]),
                int(input.shape[3] / self.patch_embed_a.patch_size[1]),
                16,
            )
        elif modality == "v":
            target = self.patchify(
                input,
                3,
                int(input.shape[2] / self.patch_embed_v.patch_size[0]),
                int(input.shape[3] / self.patch_embed_v.patch_size[1]),
                16,
            )

        # patch-wise normalization might minorly improve the classification performance, but will make the model lose inpainting function
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        audio,
        imgs,
        mask_ratio_a=0.75,
        mask_ratio_v=0.75,
        mae_loss_weight=1.0,
        contrast_loss_weight=0.01,
        mask_mode="unstructured",
    ):
        # latent is used for reconstruction (mae), latent_c_{a,v} are used for contrastive learning
        (
            latent,
            mask_a,
            ids_restore_a,
            mask_v,
            ids_restore_v,
            latent_c_a,
            latent_c_v,
        ) = self.forward_encoder(
            audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode
        )
        # if mae loss is used
        if mae_loss_weight != 0:
            pred_a, pred_v = self.forward_decoder(
                latent, mask_a, ids_restore_a, mask_v, ids_restore_v
            )
            loss_mae_a = self.forward_mae_loss(audio, pred_a, mask_a, "a")
            loss_mae_v = self.forward_mae_loss(imgs, pred_v, mask_v, "v")
            loss_mae = mae_loss_weight * (loss_mae_a + loss_mae_v)
        else:
            loss_mae_a, loss_mae_v, loss_mae = (
                torch.tensor(0.0, device=audio.device),
                torch.tensor(0.0, device=audio.device),
                torch.tensor(0.0, device=audio.device),
            )

        # if contrastive loss is used
        if contrast_loss_weight != 0:
            # note this is single directional
            loss_c, c_acc = self.forward_contrastive(
                latent_c_a.mean(dim=1), latent_c_v.mean(dim=1)
            )
            loss_c = contrast_loss_weight * loss_c
        else:
            loss_c, c_acc = torch.tensor(0.0, device=audio.device), torch.tensor(
                0.0, device=audio.device
            )

        loss = loss_mae + loss_c

        return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc

    # used only for inpainting, ignore if inpainting is not of interest
    def forward_inpaint(
        self,
        audio,
        imgs,
        mask_ratio_a=0.75,
        mask_ratio_v=0.75,
        mask_mode="unstructured",
    ):
        (
            latent,
            mask_a,
            ids_restore_a,
            mask_v,
            ids_restore_v,
            latent_c_a,
            latent_c_v,
        ) = self.forward_encoder(
            audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode
        )
        pred_a, pred_v = self.forward_decoder(
            latent, mask_a, ids_restore_a, mask_v, ids_restore_v
        )  # [N, L, p*p*3]
        loss_pixel_a = self.forward_mae_loss(audio, pred_a, mask_a, "a")
        loss_pixel_v = self.forward_mae_loss(imgs, pred_v, mask_v, "v")
        return pred_a, pred_v, mask_a, mask_v, loss_pixel_a, loss_pixel_v

    # used for retrieval, ignore if retrieval is not of interest
    def forward_feat(self, a, v):
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # the modality-specific stream
        for blk in self.blocks_a:
            a = blk(a)

        for blk in self.blocks_v:
            v = blk(v)

        # use modality specific normalization,
        for blk in self.blocks_u:
            a = blk(a, "a")
        a = self.norm_a(a)

        for blk in self.blocks_u:
            v = blk(v, "v")
        v = self.norm_v(v)
        return a, v


# the finetuned CAV-MAE model
class CAVMAEFT(nn.Module):
    def __init__(
        self,
        label_dim,
        img_size=224,
        audio_length=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        modality_specific_depth=11,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        tr_pos=True,
    ):
        super().__init__()
        timm.models.vision_transformer.Block = Block
        print("Use norm_pix_loss: ", norm_pix_loss)

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print(
            "Number of Audio Patches: {:d}, Visual Patches: {:d}".format(
                self.patch_embed_a.num_patches, self.patch_embed_v.num_patches
            )
        )

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(
            torch.zeros(1, self.patch_embed_a.num_patches, embed_dim),
            requires_grad=tr_pos,
        )  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(
            torch.zeros(1, self.patch_embed_v.num_patches, embed_dim),
            requires_grad=tr_pos,
        )  # fixed sin-cos embedding

        self.blocks_a = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(modality_specific_depth)
            ]
        )
        self.blocks_v = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(modality_specific_depth)
            ]
        )
        self.blocks_u = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(12 - modality_specific_depth)
            ]
        )

        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim)
        )

        self.initialize_weights()

        print("Audio Positional Embedding Shape:", self.pos_embed_a.shape)
        print("Visual Positional Embedding Shape:", self.pos_embed_v.shape)

    def get_patch_num(self, input_shape, stride):
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_proj = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
        test_output = test_proj(test_input)
        print(test_output.shape)
        return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

    def initialize_weights(self):
        pos_embed_a = get_2d_sincos_pos_embed(
            self.pos_embed_a.shape[-1],
            8,
            int(self.patch_embed_a.num_patches / 8),
            cls_token=False,
        )
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(
            self.pos_embed_v.shape[-1],
            int(self.patch_embed_v.num_patches**0.5),
            int(self.patch_embed_v.num_patches**0.5),
            cls_token=False,
        )
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a, std=0.02)
        torch.nn.init.normal_(self.modality_v, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, a, v, mode):
        # multi-modal fine-tuning, our default method for fine-tuning
        if mode == "multimodal":
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_v:
                v = blk(v)

            x = torch.cat((a, v), dim=1)

            for blk in self.blocks_u:
                x = blk(x)
            x = self.norm(x)

            x = x.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # finetune with only audio (and inference with only audio when the model is finetuned with only audio)
        elif mode == "audioonly":
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # note here uses the 'a' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                a = blk(a, "a")
            a = self.norm_a(a)
            x = a.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # finetune with only image (and inference with only audio when the model is finetuned with only image)
        elif mode == "videoonly":
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # note here uses the 'v' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                v = blk(v, "v")
            v = self.norm_v(v)
            x = v.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # used in case that the model is finetuned with both modality, but in inference only audio is given
        elif mode == "missingaudioonly":
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = a
            for blk in self.blocks_u:
                u = blk(u)  # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                a = blk(a, "a")  # note here use modality-specific normalization
            a = self.norm_a(a)
            a = a.mean(dim=1)

            # average the output of the two forward passes
            x = (u + a) / 2
            x = self.mlp_head(x)
            return x

        # used in case that the model is fine-tuned with both modality, but in inference only image is given
        elif mode == "missingvideoonly":
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = v
            for blk in self.blocks_u:
                u = blk(u)  # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                v = blk(v, "v")  # note here use modality-specific normalization
            v = self.norm_v(v)
            v = v.mean(dim=1)

            # average the output of the two forward passes
            x = (u + v) / 2
            x = self.mlp_head(x)
            return x

    # for retrieval
    def forward_feat(self, a, v, mode="av"):
        # return both audio and visual
        if mode == "av":
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_v:
                v = blk(v)

            for blk in self.blocks_u:
                a = blk(a, "a")
            a = self.norm_a(a)

            for blk in self.blocks_u:
                v = blk(v, "v")

            v = self.norm_v(v)
            return a, v

        # return only audio
        if mode == "a":
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_u:
                a = blk(a, "a")

            a = self.norm_a(a)
            return a
