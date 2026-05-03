import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoBackbone(nn.Module):
    """Returns (l1, l2, l3, l4) at strides (4, 8, 16, 32) with channels (256, 512, 1024, 2048)."""

    HUB_MODELS = {
        'dinov2b': 'dinov2_vitb14_reg',
        'dinov2l': 'dinov2_vitl14_reg',
    }

    def __init__(self, backbone='dinov2b', pretrained=True,
                 out_channels=(256, 512, 1024, 2048), out_strides=(4, 8, 16, 32), block_indices=None):
        super().__init__()
   
        self.vit = torch.hub.load('facebookresearch/dinov2', self.HUB_MODELS[backbone])                                                                                                                    
        print(f"Using pretrained DINO weights from hub: {self.HUB_MODELS[backbone]}")

            

        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = self.patch_size[0]

        n = len(self.vit.blocks)
        if block_indices is None:
            block_indices = (n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1)
        self.block_indices = list(block_indices)

        self.necks = nn.ModuleList([nn.Conv2d(self.embed_dim, c, 1) for c in out_channels])
        self.out_strides = out_strides
        self.out_channels = out_channels

    def forward(self, x):
        B, _, H, W = x.shape
        ps = self.patch_size
        ph = ((H + ps - 1) // ps) * ps
        pw = ((W + ps - 1) // ps) * ps
        x_in = F.interpolate(x, (ph, pw), mode='bilinear', align_corners=False) if (ph, pw) != (H, W) else x

        feats = self.vit.get_intermediate_layers(
            x_in, n=self.block_indices, reshape=True, norm=True
        )

        outs = []
        for i, feat in enumerate(feats):
            feat = self.necks[i](feat)
            th = H // self.out_strides[i]
            tw = W // self.out_strides[i]
            feat = F.interpolate(feat, (th, tw), mode='bilinear', align_corners=False)
            outs.append(feat)
        return outs
