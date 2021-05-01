from Datasets import NuScenes

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

camera_shape = [[1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [-1, -1, 1, 1, -1, 0, -1, 0, 1, 0, 1], [-1, 1, 1, -1, -1, 0, 1, 1.5, 1, 0, -1]]
camera_shape = np.array(camera_shape)


def show_extrinsic(r: np.ndarray, t: np.ndarray, ax_3d: plt.Axes):
    """

    :param r: 3*3
    :param t: 3*1 or 3
    :param ax_3d:
    :return: ax_3d
    """
    if t.ndim == 1:
        t = t.reshape((3, 1))

    cam_transported = (r @ camera_shape) + t
    ax_3d.plot(*list(cam_transported))
    return ax_3d


if __name__ == '__main__':
    ax: plt.Axes = plt.subplot(1, 2, 1, projection='3d')
    ax_img: plt.Axes = plt.subplot(1, 2, 2)
    ax.set_xlim(935, 1015)
    ax.set_ylim(610, 690)
    ax.set_zlim(-40, 40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    all_trans = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor()
    ])

    NS_ROOT = 'E:\\nuScenes'
    dataset = NuScenes.NS_Single_ImgPose_Dataset(NS_ROOT, all_trans, scene_id=0)
    data_loader = DataLoader(dataset, batch_size=1)
    for i, d in enumerate(data_loader):
        img, rot, tra = d
        # print(rot, tra)
        # print(rot[0] @ rot[0].T)

        if i % 20 == 0:
            ax = show_extrinsic(rot[0], tra[0], ax)
            ax_img.imshow(img[0].permute(1, 2, 0))
            plt.draw()
            plt.pause(0.5)
            ax_img.clear()

        if i > 1000:
            break
    plt.show()
