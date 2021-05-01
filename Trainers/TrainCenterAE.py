from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm

from torchvision.utils import save_image

from Datasets import OpenImages
from Models import CenterAE

from itertools import chain


def train_general(train_loader, start_epoch=0, start_iter=0):
    enc = CenterAE.Encoder().cuda()
    dec = CenterAE.DecoderCenter().cuda()

    if start_iter > 0:
        print('load from checkpoint: %08d' % start_iter)
        checkpoint_path = '../weights/center/dec_%04d_%08d.pth' % (start_epoch, start_iter)
        checkpoint = torch.load(checkpoint_path)
        dec.load_state_dict(checkpoint)

    optim_enc = torch.optim.Adam(enc.parameters(), lr=3e-5)
    optim_dec = torch.optim.Adam(dec.parameters(), lr=1e-5)
    loss_func = torch.nn.MSELoss()
    for epoch in range(start_epoch, 10):
        print('epoch: %04d' % epoch)
        data_tq = tqdm(train_loader)
        for i, data in enumerate(data_tq):
            img, coord, img_coord = data
            img = img.cuda()
            coord = coord.cuda()
            img_coord = img_coord.cuda()

            # data = data.cuda()

            if i % 10 == 0:
                enc.train()
                code = enc.forward(img, grad=True)
            else:
                code = enc.forward(img, grad=False)

            img_recover = dec(code, coord)

            loss = loss_func(img_recover, img_coord)

            optim_dec.zero_grad()
            loss.backward()
            optim_dec.step()

            if i % 10 == 0:
                optim_enc.zero_grad()
                optim_enc.step()
                enc.eval()

            total_iter = start_iter + i + 1
            data_tq.set_description('iter: %d loss: %f' % (total_iter, loss.detach().cpu().numpy()))
            if total_iter % 100 == 0:
                d_img = torch.cat([img[0], img[1]], dim=1)
                with torch.no_grad():
                    recover = dec.decode_full(code[0:2])
                i_img = torch.cat([recover[0], recover[1]], dim=1)
                all_img = torch.cat([d_img, i_img], dim=2)
                save_image(all_img, '../img_results/center/img_%04d_%08d.png' % (epoch, total_iter))
            if total_iter % 1000 == 0:
                torch.save(dec.state_dict(), '../weights/center/dec_%04d_%08d.pth' % (epoch, total_iter))


if __name__ == '__main__':
    DATA_ROOT = 'F:\\Datasets\\OpenImages'
    START_ITER = 10000
    START_EPOCH = 3

    all_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = OpenImages.Train0_Coord(data_root=DATA_ROOT, transform=all_trans)
    data_loader = DataLoader(dataset, batch_size=8, num_workers=16, shuffle=True)

    train_general(data_loader, START_EPOCH, START_ITER)
