from torch.utils.data import Dataset

import json
import os

from PIL import Image
import numpy as np

from utils import GeometricUtils


class NS_Seq_ImgPose_Dataset(Dataset):
    """
    This dataset will load image sequence (with the expected length) along with camera pose.
    """
    def __init__(self, data_root, seq_length):
        scenes_meta_path = os.path.join(data_root, 'v1.0-trainval_meta', 'v1.0-trainval', 'scene.json')
        with open(scenes_meta_path) as f:
            scenes_meta = json.load(f)

        self.seq_length = seq_length
        # data_list contains available seq_id and start_frame
        self.data_list = []

    def __getitem__(self, item):
        return

    def __len__(self):
        return len(self.data_list)


class NS_Single_ImgPose_Dataset(Dataset):
    """
    This dataset will load single image and its pose.
    All frames are from the same scene.
    """

    def __init__(self, data_root, trans, *, scene_name=None, scene_id=None):
        meta_path = os.path.join(data_root, 'v1.0-trainval_meta', 'v1.0-trainval')
        if scene_name is None:
            scenes_meta_path = os.path.join(meta_path, 'scene.json')
            with open(scenes_meta_path) as f:
                scenes_meta = json.load(f)
                scene_name = scenes_meta[scene_id]['name']
                print('NS_Single_ImgPose_Dataset: Loading from ' + scene_name)
                del scenes_meta

        self.scene_name = scene_name
        self.img_root = os.path.join(data_root, 'v1.0-trainval')
        self.data_list = []
        for view in ('FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT'):
            scenes_meta_path = os.path.join(data_root, 'my_anns', 'ImgSeq_Lists', scene_name, 'CAM_%s.json' % view)
            with open(scenes_meta_path) as f:
                scenes_meta = json.load(f)
                self.data_list += [[meta['filename'], meta['ego_pose_token']] for meta in scenes_meta]
                del scenes_meta

        calibrated_sensor_path = os.path.join(meta_path, 'calibrated_sensor.json')
        with open(calibrated_sensor_path) as f:
            calibrated_sensor = json.load(f)
            calibrated_sensor = {cs['token']: [cs['rotation'], cs['translation']] for cs in calibrated_sensor}

        ego_pose_path = os.path.join(meta_path, 'ego_pose.json')
        with open(ego_pose_path) as f:
            ego_poses = json.load(f)
            # rot = GeometricUtils.quaternion_2_rotation(ego_pose_dir[0])
            self.ego_pose_dir = {pose['token']: [pose['rotation'], pose['translation']] for pose in ego_poses}
            del ego_poses

        self.trans = trans

    def __getitem__(self, item):
        img_path, pose_token = self.data_list[item]
        img_path = os.path.join(self.img_root, img_path)
        img = Image.open(img_path)
        img = self.trans(img)
        ego_pose_dir = self.ego_pose_dir[pose_token]
        rot = GeometricUtils.quaternion_2_rotation(ego_pose_dir[0])
        return img, np.array(rot), np.array(ego_pose_dir[1])  # rgb, rotation, translation

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import transforms
    from torch.utils.data.dataloader import DataLoader

    all_trans = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor()
    ])

    NS_ROOT = 'E:\\nuScenes'
    dataset = NS_Single_ImgPose_Dataset(NS_ROOT, all_trans, scene_id=0)
    data_loader = DataLoader(dataset, batch_size=4)
    for d in data_loader:
        print(d[1])
        plt.imshow(d[0][0].permute(1, 2, 0).numpy())
        plt.show()
        break
