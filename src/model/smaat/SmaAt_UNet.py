from torch import nn
from src.model.smaat.unet_parts import OutConv
from src.model.smaat.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from src.model.smaat.layers import CBAM


class SmaAt_UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        # ours
        # ---

        # 1. naive
        # - [B, C=11, T=4 , H, W] -> [B, C=1, T=16, H, W]
        # - [B, C=1,  T=16, H, W] -> [B, 1]
            # regression target ^^

        # 2. sum and average y over time dim (T)
        # - [B, C=11, T=4 , H, W] -> [B, C=1, T=1 , H, W]
        # - [B, C=1 , T=1 , H, W] -> [B, 1]

        # 3. sum and average X over time dim (T)
        # - [B, C=11, T=1 , H, W] -> [B, C=1, T=16, H, W]
        # - [B, C=1,  T=16, H, W] -> [B, 1]

        # 4. combine: 2+3
        # - [B, C=11, T=1 , H, W] -> [B, C=1, T=1 , H, W]
        # - [B, C=1,  T=1 , H, W] -> [B, 1]

        # theirs
        # ---
        # [B, C=12, T=1, H, W] -> [B, C=1, T=1, H, W]
        
        # conv 2d layers
        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1   = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2   = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3   = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4   = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

        # NOTE: ours; a little regression head
        self.linear = nn.Linear(252 ** 2, 1)

    def forward(self, x):

        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)

        # [B, C=64, H, W] -> [B, C=1, H, W]
        x: torch.Tensor = self.outc(x)
        
        # [B, C=1, H, W] -> [B, H * W]
        x = x.squeeze(0)
        x = x.flatten(1, 2)

        # -> [B]
        x = self.linear(x).squeeze(1)

        return x


if __name__ == "__main__":
    
    import torch
    net = SmaAt_UNet(n_channels=11, n_classes=1)
    X   = torch.rand(1, 11, 252, 252)
    x = net(X)
    print(x.shape)
