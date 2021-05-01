from torch import nn
# from torch.nn import functional as F
import memcnn
import torch
from efficientnet_pytorch import EfficientNet


class Encoder(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        if pretrained is None:
            self.encoder = EfficientNet.from_pretrained('efficientnet-b4')
        else:
            print('load from ' + pretrained)
            self.encoder = EfficientNet.from_name('efficientnet-b4')
            self.encoder.load_state_dict(torch.load(pretrained))

    def forward(self, x, grad=False):
        if not grad:
            with torch.no_grad():
                feat = self.encoder.extract_features(x)
        else:
            feat = self.encoder.extract_features(x)
        return feat

    def save(self, path):
        torch.save(self.encoder.state_dict(), path)


class HyperNetCell(nn.Module):
    def __init__(self, in_channel: int = 64, out_channel: int = 64, feature_channel: int = 768,
                 feature_next: bool = True):
        super().__init__()

        self.in_channel: int = in_channel
        self.out_channel: int = out_channel
        self.feature_next: bool = feature_next

        self.feature_net: nn.Module = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(feature_channel, 768, kernel_size=3, padding=1, groups=2)),
            nn.LeakyReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(768, 1536, kernel_size=3, padding=1, groups=3)),
            nn.LeakyReLU(inplace=True),
        )  # feature for the weight net
        if feature_next:
            self.feature_net_next: nn.Module = nn.Sequential(
                nn.Conv2d(feature_channel, feature_channel, kernel_size=3, padding=1, groups=3),
                nn.LeakyReLU(inplace=True),
                nn.utils.weight_norm(nn.Conv2d(feature_channel, feature_channel, kernel_size=3, padding=1, groups=2)),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(feature_channel, feature_channel, kernel_size=3, padding=1, groups=3),
            )  # feature for next cell

        self.feature2weight: nn.Module = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1536, 1024)),
            nn.LeakyReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.LeakyReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, in_channel * out_channel),
        )

    def forward(self, feature):
        batch = feature.shape[0]
        # batch, 768, h, w
        feature_w = self.feature_net(feature).mean(dim=(2, 3))
        # batch, 768
        weight = self.feature2weight(feature_w)
        weight = weight.view(batch, self.in_channel, self.out_channel)
        # batch, in_channel, out_channel
        if self.feature_next:
            feature_n = self.feature_net_next(feature)
            return weight, feature_n + feature
        else:
            return weight


class LinearResBlock(nn.Module):
    def __init__(self, channel=64, n=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channel, channel*n),
            nn.ReLU(inplace=True),
            nn.Linear(channel*n, channel*n),
            nn.ReLU(inplace=True),
            nn.Linear(channel*n, channel),
        )

    def forward(self, x):
        y = self.net(x)
        return x+y


class MemLinear(nn.Module):
    def __init__(self, channel=32, n=4):
        super().__init__()

        invertible_module = memcnn.AdditiveCoupling(
            Fm=nn.Sequential(
                nn.utils.weight_norm(nn.Linear(channel, channel*n)),
                nn.LeakyReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(channel*n, channel*n)),
                nn.LeakyReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(channel*n, channel*n)),
                nn.LeakyReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(channel*n, channel)),
            ),
            Gm=nn.Sequential(
                nn.utils.weight_norm(nn.Linear(channel, channel*n)),
                nn.LeakyReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(channel*n, channel*n)),
                nn.LeakyReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(channel*n, channel*n)),
                nn.LeakyReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(channel*n, channel)),
                nn.LeakyReLU(inplace=True),
            )
        )
        self.model = memcnn.InvertibleModuleWrapper(fn=invertible_module)

    def forward(self, x):
        batch, n_points, channel = x.shape
        return self.model(x.view(batch*n_points, channel)).view(batch, n_points, channel)


class HyperNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = 12

        in_channels = [64, 64, 64, 96, 64, 64, 64, 96, 64, 64, 64, 32]
        self.weight_layers = nn.ModuleList([HyperNetCell(in_channel=in_channels[i]) for i in range(self.layers-1)])
        self.weight_layers.append(HyperNetCell(in_channel=in_channels[-1], feature_next=False))  # last layer

        n_lists = [2, 3, 2, 3, 2, 3, 2, 3, 4, 4, 4, 4]
        self.mapping_layers = nn.ModuleList([LinearResBlock(n=n_lists[i]) for i in range(self.layers)])

        self.rgb_generator = nn.Sequential(
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def point_forward(self, points, weights):
        # (batch, n_points, in_channel) @ (batch, in_channel, out_channel) -> (batch, n_points, out_channel)
        batch, n_points, _ = points.shape
        point_middle = torch.tensor([[[]]]).view(batch, n_points, 0).cuda()
        for i in range(self.layers):
            if i % 4 == 0:
                point_middle = torch.cat([points, point_middle], dim=2)
            point_middle = self.relu(torch.matmul(point_middle, weights[i]))
            point_middle = self.relu(self.mapping_layers[i](point_middle))
        return point_middle

    def weights_forward(self, features):
        weights = []
        for i in range(self.layers-1):
            w, feature = self.weight_layers[i](features)
            weights.append(w)
        weights.append(self.weight_layers[self.layers-1](features))
        weights.reverse()
        return weights

    def forward(self, features, points):
        weights = self.weights_forward(features)
        point_middle = self.point_forward(points, weights)

        # (batch, n_points, 3) - > (batch, 3, n_points)
        rgb = self.rgb_generator(point_middle).permute(0, 2, 1)

        return rgb * 1.1 - 0.05

    def forward_point_batch(self, features, points, point_batch_size, point_total):
        # ===== IMAGE BATCH = 1 =====
        # The "batch" of this function means "batch of points".
        # f1 shape = (1, channel, h, w)
        # points shape = (1, 32, 65536)

        weights = self.weights_forward(features)

        img_buffer = []
        coord_i = 0
        while coord_i < point_total:
            # weights.shape = (bat, 32, 32)
            # ===== NO BATCH OF IMAGE! =====
            # coo_batch.shape = (1, coord_batch, 32)
            # middle.shape = (1, coord_batch, 32) @ (1, 32, 32) -> (1, coord_batch, 32)
            coo_batch = points[:, coord_i: coord_i + point_batch_size]

            # (batch, n_points, in_channel) @ (batch, in_channel, out_channel) -> (batch, n_points, out_channel)
            point_middle = self.point_forward(coo_batch, weights)

            # (batch, n_points, 3) - > (3, n_points)
            rgb = self.rgb_generator(point_middle)[0].T

            img_buffer.append(rgb)

            coord_i += point_batch_size
        img = torch.cat(img_buffer, dim=1).view(1, 3, 256, 256)

        return img * 1.1 - 0.05


class DecoderCenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1792, 768, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3, padding=1, groups=3),
            nn.LeakyReLU(inplace=True),  # 8 * 8
        )

        self.hyper_net = HyperNet()

        self.coord_batch = 256 * 64
        self.coord_total = 256 * 256

        r = torch.arange(-128, 128) / 128
        grid = torch.meshgrid([r, r])
        img_grid = torch.cat([grid[1], grid[0]]).view(2, 256, 256).repeat(16, 1, 1)
        img_grid = img_grid * torch.exp2((torch.arange(0, 16)/2)[:, None].repeat(1, 1, 2).view(32, 1, 1))
        img_grid = torch.sin(img_grid)
        # img_grid.shape = (256*256, 32)
        self.img_grid = img_grid.view(32, 256*256).T.cuda()

    def decode_full(self, feature):
        batch = feature.shape[0]
        f1 = self.conv_net(feature)

        final_images = []
        for bat in range(batch):
            final_img = self.hyper_net.forward_point_batch(f1[bat:bat+1], self.img_grid[None],
                                                           self.coord_batch, self.coord_total)
            final_images.append(final_img)

        final_images = torch.cat(final_images)
        return final_images

    def forward(self, feature, points):
        """
        :param feature:
        :param points: mapped points, shape = (batch, n_points, 32),
        :return:
        """
        batch, n_points, _ = points.shape

        f1 = self.conv_net(feature)
        # (N, C) -> (N * n_P, C)

        rgb = self.hyper_net(f1, points)

        return rgb
