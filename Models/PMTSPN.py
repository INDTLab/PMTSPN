import torch
import torch.nn as nn
from Models import Conformer, SwinTransformer


def bilinear(x):
    '''
      input: [N, C, H, W]
      output: [N, C, C]
      Bilinear pooling and sqrt for X-self.
    '''
    if len(x.shape) == 3:
        x = x.permute(0, 2, 1)
    if len(x.shape) == 4:
        N, C, H, W = x.shape
        x = torch.reshape(x, (N, C, H * W))
    N, C, HW = x.shape
    x = torch.bmm(x, torch.transpose(x, 1, 2)) / (HW)
    x = torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))
    return x

class PMTSPN(nn.Module):

    def __init__(self, device):
        super(multiswin, self).__init__()
        self.features = Conformer.Conformer(embed_dim=384, num_heads=6, qkv_bias=True)
        self.features.load_state_dict(torch.load("Module/Conformer_small_patch16.pth", map_location=torch.device(device)))
        dims = [64, 64, 128, 256]
        hirs = [16, 16, 16, 32]
        wirs = [16, 16, 32, 32]
        wins = [4, 4, 8, 16]
        self.trans1 = nn.ModuleList([nn.Sequential(nn.AvgPool1d(8, 8),
                                                   *[SwinTransformer.BasicLayer(dim=dims[n], input_resolution=(
                                                   hirs[n] // 2 ** i, wirs[n] // 2 ** i), depth=2, num_heads=8,
                                                                                window_size=wins[n] // 2 ** i,
                                                                                mlp_ratio=4., qkv_bias=True,
                                                                                qk_scale=None, drop=0., attn_drop=0.,
                                                                                drop_path=0., norm_layer=nn.LayerNorm,
                                                                                downsample=SwinTransformer.PatchMerging,
                                                                                use_checkpoint=False) for i in
                                                     range(3)]) for n in range(4)])

        self.trans2 = nn.ModuleList([nn.Sequential(nn.AvgPool1d(3, 3),
                                                   *[SwinTransformer.BasicLayer(dim=256, input_resolution=(
                                                       16 // 2 ** i, 24 // 2 ** i), depth=2, num_heads=8,
                                                                                window_size=8 // 2 ** i,
                                                                                mlp_ratio=4., qkv_bias=True,
                                                                                qk_scale=None, drop=0., attn_drop=0.,
                                                                                drop_path=0., norm_layer=nn.LayerNorm,
                                                                                downsample=SwinTransformer.PatchMerging,
                                                                                use_checkpoint=False) for i in
                                                     range(3)]) for n in range(4)])
        self.final_learner = nn.Sequential(
            nn.Linear(11776, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        xs1, xts1 = self.features(x1)
        xs2, xts2 = self.features(x2)
        # 1 4 8 12[0,3,7,11]
        flags = [0, 3, 7, 11]
        res = []
        rets = []
        for i, f in enumerate(flags):
            x1, xt1, x2, xt2 = bilinear(xs1[f]), bilinear(xts1[f]), bilinear(xs2[f]), bilinear(xts2[f])
            x1 = torch.cat((x1, x2), dim=2)
            x2 = torch.cat((xt1, xt2), dim=2)
            x1 = self.trans1[i](x1).flatten(1)
            x2 = self.trans2[i](x2).flatten(1)
            res.append(x1)
            rets.append(x2)

        res = torch.cat(res, dim=1)
        rets = torch.cat(rets, dim=1)
        result = torch.cat((res, rets), dim=1)
        return self.final_learner(result)
