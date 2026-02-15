import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        
        x = x.permute(0, 2, 3, 1) 
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

class ModernBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 

        self.norm = LayerNorm2d(dim)
        
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()

        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.drop_prob = drop_path

    def forward(self, x):
        input = x 

        x = self.dwconv(x)
        x = self.norm(x)

        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        x = input + x 
        return x

class ModernCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, depths=[3, 3, 3], dims=[64, 128, 256]):
        
        super().__init__()
        
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        for i in range(len(dims)-1):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList() 
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[ModernBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.mean([-2, -1]) 
        
        x = self.norm(x)
        x = self.head(x)
        return x