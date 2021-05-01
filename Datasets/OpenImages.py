from torch.utils.data import Dataset
import torch
from torch.nn import functional as F

import os
from PIL import Image

from torchvision import transforms


class Train(Dataset):
    def __init__(self, data_root, group, transform):
        # 'D:\Datasets\OpenImages'
        self.data_root = data_root
        self.transform = transform

        self.group = group

        with open(os.path.join(self.data_root, 'image_lists', 'train_%d.txt' % self.group)) as f:
            self.data_list = f.read().split()
        # print(len(self.data_list))

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.data_root, 'images', 'train_%d' % self.group, self.data_list[item])).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data_list)


class Train_Coord(Dataset):
    def __init__(self, data_root, group, transform):
        # 'D:\Datasets\OpenImages'
        self.data_root = data_root
        self.transform = transform
        self.n_point = 16384
        self.tot = transforms.ToTensor()

        self.group = group

        with open(os.path.join(self.data_root, 'image_lists', 'train_%d.txt' % self.group)) as f:
            self.data_list = f.read().split()
        # print(len(self.data_list))

    def __getitem__(self, item):
        """
        img: (3, H, W)
        coord: (n_points, 2)
        img_coord: (3, n_points)
        """
        img_path = os.path.join(self.data_root, 'images', 'train_%d' % self.group, self.data_list[item])
        img_0 = Image.open(img_path).convert('RGB')
        img = self.transform(img_0)
        # coord = torch.randn(self.n_point, 2) / 3
        # coord = coord.clip(-1, 1)
        coord = torch.arcsin((torch.rand(self.n_point, 2) - 0.5) * 2) / 1.5
        coord = coord.clip(-1, 1)
        # sample image points from original resolution
        img_coord = F.grid_sample(self.tot(img_0)[None], coord[None, None, :, :])  # img_coord: (1, 3, 1, n_points)

        # (n_P, 2) -> (n_P, 16 * 2) -> (n_P, 16, 2)
        coord_mapped = coord.repeat(1, 16).view(self.n_point, 16, 2)
        # (n_P, 16, 2) * (1, 16, 2) -> (n_P, 16, 2)
        coord_mapped = coord_mapped * torch.exp2((torch.arange(0, 16) / 2)[:, None].repeat(1, 2))[None]
        # (n_P, 32)
        coord_mapped = torch.sin(coord_mapped.view(self.n_point, 32))

        return img, coord_mapped, img_coord.view(3, self.n_point)

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data.dataloader import DataLoader
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    DATA_ROOT = 'F:\\Datasets\\OpenImages'

    all_trans = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor()
    ])
    dataset = Train_Coord(data_root=DATA_ROOT, group=0, transform=all_trans)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=16, shuffle=True)
    counter = 0
    for a in tqdm(data_loader):
        img, coord, img_coord = a
        y, x = coord[0].chunk(2, -1)
        plt.plot(y, x, 'o', alpha=0.2, markersize=4)
        plt.show()
