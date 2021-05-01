from torch import nn
from torch.nn import functional as F
import torch
from efficientnet_pytorch import EfficientNet

import memcnn
from itertools import chain


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained('efficientnet-b4')

    def forward(self, x, grad=False):
        if not grad:
            with torch.no_grad():
                feat = self.encoder.extract_features(x)
        else:
            feat = self.encoder.extract_features(x)
        return feat


class MemLinear(nn.Module):
    def __init__(self, channel):
        super().__init__()

        invertible_module = memcnn.AdditiveCoupling(
            Fm=nn.Sequential(
                nn.utils.weight_norm(nn.Linear(channel, channel)),
                nn.LeakyReLU(inplace=True),
            ),
            Gm=nn.Sequential(
                nn.utils.weight_norm(nn.Linear(channel, channel)),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.model = memcnn.InvertibleModuleWrapper(fn=invertible_module)

    def forward(self, x):
        return self.model(x)


class DecoderCenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1792, 768, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3, padding=1, groups=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3, padding=1, groups=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3, padding=0, groups=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(768, 128, kernel_size=3, padding=0, groups=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),  # 128 * 4 * 4
        )

        self.linear_net_1 = nn.Sequential(
            nn.Linear(2048+32, 1024),
            nn.LeakyReLU(),
            MemLinear(512),
            MemLinear(512),
            MemLinear(512),
            MemLinear(512),
            MemLinear(512),
            MemLinear(512),
            MemLinear(512),
            MemLinear(512),
        )
        self.linear_net_2 = nn.Sequential(
            nn.Linear(1024+32, 512),
            nn.LeakyReLU(),
            MemLinear(256),
            MemLinear(256),
            MemLinear(256),
            MemLinear(256),
            MemLinear(256),
            MemLinear(256),
            MemLinear(256),
            MemLinear(256),
            nn.Linear(512, 3),
            nn.Sigmoid(),
        )
        self.feat_channel = 128 * 4 * 4
        self.coord_batch = 256 * 64
        self.coord_total = 256 * 256

        r = torch.arange(-128, 128) / 128
        grid = torch.meshgrid([r, r])
        img_grid = torch.cat([grid[1], grid[0]]).view(2, 256, 256).repeat(16, 1, 1)
        img_grid = img_grid * torch.exp2((torch.arange(0, 16)/2)[:, None].repeat(1, 1, 2).view(32, 1, 1))
        img_grid = torch.sin(img_grid)
        self.img_grid = img_grid.view(32, 256*256).T.cuda()

    def decode_full(self, feature):
        batch = feature.shape[0]
        feature = self.conv_net(feature)

        final_imgs = []
        for bat in range(batch):
            img_buffer = []
            coord_i = 0
            feature_bat = feature[None, bat].repeat(self.coord_batch, 1)
            while coord_i < self.coord_total:
                coo_batch = self.img_grid[coord_i: coord_i + self.coord_batch]
                middle = self.linear_net_1(torch.cat([coo_batch, feature_bat], dim=1))
                rgb = self.linear_net_2(torch.cat([coo_batch, middle], dim=1))
                img_buffer.append(rgb.T)
                coord_i += self.coord_batch
            img = torch.cat(img_buffer, dim=1).view(3, 256, 256)
            final_imgs.append(img[None])
        final_imgs = torch.cat(final_imgs, dim=0)
        return final_imgs * 1.1 - 0.05

    def forward(self, feature, points):
        """
        :param feature:
        :param points: shape = (batch, n-points, 2), range = (-1, 1)
        :return:
        """
        batch, n_points, _ = points.shape

        # points = torch.randn(self.coord_batch, 1, 2).repeat(1, 16, 1)
        # (N, n_P, 2) -> (N, n_P, 16 * 2) -> (N * n_P, 16, 2)
        points_batch = points.repeat(1, 1, 16).view(batch*n_points, 16, 2)
        # (N * n_P, 16, 2) * (1, 16, 2) -> (N * n_P, 16, 2)
        points_batch = points_batch * torch.exp2((torch.arange(0, 16)/2)[:, None].repeat(1, 2))[None].cuda()
        # (N * n_P, 32)
        points_batch = torch.sin(points_batch.view(batch*n_points, 32))

        feature = self.conv_net(feature)
        # (N, C) -> (N * n_P, C)
        feature_bat = feature[:, None].repeat(1, n_points, 1).view(batch*n_points, self.feat_channel)

        middle = self.linear_net_1(torch.cat([points_batch, feature_bat], dim=1))
        rgb = self.linear_net_2(torch.cat([points_batch, middle], dim=1))
        # (N * n_P, C) -> (N, n_P, C) -> (N, C, n_P)
        rgb = rgb.view(batch, n_points, 3).permute(0, 2, 1)

        return rgb * 1.1 - 0.05
