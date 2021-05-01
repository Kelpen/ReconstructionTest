from torch.utils.data import Dataset
import json
from PIL import Image
import os.path
import torch


class YTVOS_Anno_2019(Dataset):
    """
    The loader will return a sequence of images.
        The sequence contains a key-frame with annotation. There are num_before frames before the key-frame and
        num_after frames after the key frames.
    """
    def __init__(self, num_before, num_after):
        self.data_list = []

    def __getitem__(self, item):
        return

    def __len__(self):
        return len(self.data_list)


class YTVOS_ImgSeq_2019(Dataset):
    def __init__(self, data_root, seq_len, transform):
        """
        The loader will return a sequence of images.
         :param data_root: Path to the parent of 'train_all_frames'.
         :param seq_len: Length of sequences.
        The returned shape is specified by the transform.
        """
        assert transform, print('Use ToTensor, idiot!')

        self.data_root = data_root
        self.seq_len = seq_len
        self.transform = transform

        with open(os.path.join(self.data_root, 'train_all_frames_list.json'), 'r') as all_frame_file:
            self.all_frame_dict = json.load(all_frame_file)
            del all_frame_file

        self.data_list = []
        for video_name in self.all_frame_dict:
            video_frame_list = self.all_frame_dict[video_name]
            video_len = len(video_frame_list)
            for start_frame in range(video_len - self.seq_len + 1):
                self.data_list.append([video_name, start_frame])

    def __getitem__(self, item):
        video_name, start_frame = self.data_list[item]
        frame_list = self.all_frame_dict[video_name]

        img_seq = []
        for i in range(self.seq_len):
            img_name = frame_list[start_frame + i]
            img_name = os.path.join(self.data_root, 'train_all_frames', 'JPEGImages', video_name, img_name)
            img = Image.open(img_name)
            img = self.transform(img)[:, None]  # shape = (3, 1, h, w)
            img_seq.append(img)
        img_seq = torch.cat(img_seq, dim=1)  # shape = (3, seq_len, h, w)
        return img_seq

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data.dataloader import DataLoader
    from tqdm import tqdm

    DATA_ROOT = 'F:\\Datasets\\YTVOS\\VOS2019'
    all_trans = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor()
    ])
    dataset = YTVOS_ImgSeq_2019(data_root=DATA_ROOT, seq_len=7, transform=all_trans)
    data_loader = DataLoader(dataset, batch_size=8, num_workers=16, shuffle=True)
    counter = 0
    for a in tqdm(data_loader):
        pass
