# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import pandas as pd


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, args, model_name='vit_base', attn_mode='flash_attn', global_pool=False, add_w=False, device=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_dim=None, num_classes=2):
        super(VisionTransformer, self).__init__()
        
        gradient_csv_path = '/home/project/11001932/ruilin/ijepa/dataset/ukbiobank/gradient_mapping_450.csv'
        def load_gradient():
            df = pd.read_csv(gradient_csv_path, header=None)
            gradient = torch.tensor(df.values, dtype=torch.float32)
            return gradient.unsqueeze(0)

        gradient = load_gradient().to(device, non_blocking=True)
        
        from src.helper import init_model
        self.encoder, _ = init_model(
            device=device,
            patch_size=args.patch_size, # 49
            crop_size=args.crop_size, # (450, 490)
            pred_depth=args.pred_depth, # 12
            pred_emb_dim=args.pred_emb_dim, # 384
            model_name=model_name,
            gradient_pos_embed=gradient,
            attn_mode=attn_mode,
            add_w=args.add_w,
            gradient_checkpointing=args.gradient_checkpointing)
        
        self.gradient_checkpointing = args.gradient_checkpointing

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(self.encoder.embed_dim)        
        
        self.head = nn.Linear(self.encoder.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):

        x = self.encoder(x)
        if self.global_pool:
            x = x[:, :, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            outcome = x[:, 0]

        if self.gradient_checkpointing:
            try:
                x = torch.utils.checkpoint.checkpoint(self.head, outcome, use_reentrant=False)
            except ValueError as e:
                print(1)
        else:
            x = self.head(outcome)

        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=1, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, in_chans=1, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model