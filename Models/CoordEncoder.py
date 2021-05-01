from torch import nn
import torch
from efficientnet_pytorch import EfficientNet

import memcnn
from itertools import chain


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained('efficientnet-b4').eval()

    def forward(self, x, grad=False):
        if not grad:
            with torch.no_grad():
                feat = self.encoder.extract_features(x)
        else:
            feat = self.encoder.extract_features(x)
        return feat


class Decoder(nn.Module):
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
            nn.Linear(2048+16, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 512),
        )
        self.linear_net_2 = nn.Sequential(
            nn.Linear(512+16, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 3),
            nn.LeakyReLU(inplace=True),
        )
        self.coord_batch = 4096
        self.coord_total = 256 * 256

        r = torch.range(-128, 127) * 3.14159265358979323846264 / 256
        img_grid = torch.cat(torch.meshgrid([r, r])).view(2, 256, 256).repeat(8, 1, 1)
        img_grid = img_grid * torch.exp2(torch.range(0, 7)[:, None].repeat(1, 1, 2).view(16, 1, 1))
        img_grid = torch.sin(img_grid)
        self.img_grid = img_grid.view(16, 256*256).T.cuda()

    def forward(self, feature):
        batch = feature.shape[0]
        feature = self.conv_net(feature)

        final_imgs = []
        for bat in range(batch):
            img_buffer = []
            coord_i = 0
            feature_bat = feature[None, bat].repeat(self.coord_batch, 1)
            while coord_i < self.coord_total:
                coo_batch = self.img_grid[coord_i: coord_i+self.coord_batch]
                middle = self.linear_net_1(torch.cat([coo_batch, feature_bat], dim=1))
                rgb = self.linear_net_2(torch.cat([coo_batch, middle], dim=1))
                img_buffer.append(rgb.T)
                coord_i += self.coord_batch
            img = torch.cat(img_buffer, dim=1).view(3, 256, 256)
            final_imgs.append(img[None])
        final_imgs = torch.cat(final_imgs, dim=0)
        return final_imgs


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


class MemDecoder(nn.Module):
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
        self.coord_batch = 256 * 64
        self.coord_total = 256 * 256

        r = torch.arange(-128, 128) * 3.14159265358979323846264 / 256
        img_grid = torch.cat(torch.meshgrid([r, r])).view(2, 256, 256).repeat(16, 1, 1)
        img_grid = img_grid * torch.exp2((torch.arange(0, 16)/2)[:, None].repeat(1, 1, 2).view(32, 1, 1))
        img_grid = torch.sin(img_grid)
        self.img_grid = img_grid.view(32, 256*256).T.cuda()

    def forward(self, feature):
        batch = feature.shape[0]
        feature = self.conv_net(feature)

        final_imgs = []
        for bat in range(batch):
            img_buffer = []
            coord_i = 0
            feature_bat = feature[None, bat].repeat(self.coord_batch, 1)
            while coord_i < self.coord_total:
                coo_batch = self.img_grid[coord_i: coord_i+self.coord_batch]
                middle = self.linear_net_1(torch.cat([coo_batch, feature_bat], dim=1))
                rgb = self.linear_net_2(torch.cat([coo_batch, middle], dim=1))
                img_buffer.append(rgb.T)
                coord_i += self.coord_batch
            img = torch.cat(img_buffer, dim=1).view(3, 256, 256)
            final_imgs.append(img[None])
        final_imgs = torch.cat(final_imgs, dim=0)
        return final_imgs * 1.1 - 0.05


class MemDecoderTrainer(nn.Module):
    """
    Model + Trainer
    Because backpropagation during training loop can save memory.
    """
    def __init__(self, optimizer, lr_conv, lr_line):
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
        self.coord_batch = 8192
        self.coord_total = 256 * 256

        r = torch.arange(-128, 128) * 3.14159265358979323846264 / 256
        img_grid = torch.cat(torch.meshgrid([r, r])).view(2, 256, 256).repeat(16, 1, 1)
        img_grid = img_grid * torch.exp2((torch.arange(0, 16)/2)[:, None].repeat(1, 1, 2).view(32, 1, 1))
        img_grid = torch.sin(img_grid)
        self.img_grid = img_grid.view(32, 256*256).T.cuda()

        self.optim_conv = optimizer(self.conv_net.parameters(), lr=lr_conv)
        lin_net = chain(self.linear_net_1.parameters(), self.linear_net_2.parameters())
        self.optim_line = optimizer(lin_net, lr=lr_line)

    def forward(self, feature, img, loss_func):
        batch = feature.shape[0]

        # The gradient will propagate onto 'feature', then call backward on 'feature_original' manually.
        feature_original = self.conv_net(feature)
        feature = feature_original.detach()
        feature.requires_grad = True

        total_loss = 0

        for bat in range(batch):
            img_flatten = img[bat].view(3, 256*256)
            feature_bat = feature[None, bat].repeat(self.coord_batch, 1)

            coord_i = 0
            while coord_i < self.coord_total:
                coo_batch = self.img_grid[coord_i: coord_i+self.coord_batch]
                middle = self.linear_net_1(torch.cat([coo_batch, feature_bat], dim=1))
                rgb = self.linear_net_2(torch.cat([coo_batch, middle], dim=1))
                loss = loss_func(img_flatten[:, coord_i: coord_i+self.coord_batch], rgb.T)

                self.optim_line.zero_grad()
                loss.backward()
                total_loss += loss.detach().cpu()
                self.optim_line.step()

                coord_i += self.coord_batch

            # The gradient is propagated image by image.
        self.optim_conv.zero_grad()
        feature_original.backward(feature.grad)
        self.optim_conv.step()
        feature.grad = None  # clean the gradient manually

        return total_loss

    def test(self, feature):
        # single image, batch=1
        img_buffer = []

        with torch.no_grad():
            coord_i = 0
            feature = self.conv_net(feature)
            feature_bat = feature[None, 0].repeat(self.coord_batch, 1)
            while coord_i < self.coord_total:
                coo_batch = self.img_grid[coord_i: coord_i + self.coord_batch]
                middle = self.linear_net_1(torch.cat([coo_batch, feature_bat], dim=1))
                rgb = self.linear_net_2(torch.cat([coo_batch, middle], dim=1))
                img_buffer.append(rgb.T)
            img = torch.cat(img_buffer, dim=1).view(3, 256, 256)
        return img
