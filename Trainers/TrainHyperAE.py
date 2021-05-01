from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm

from torchvision.utils import save_image

from Datasets import OpenImages
from Models import HyperNetCoordAE

import os
import warnings

warnings.simplefilter("ignore")


def train(train_loader, start_epoch=0, start_iter=0, pretrained_enc=False):
    # encoder
    if pretrained_enc:
        checkpoint_path = '../weights/hyper/enc_%04d_%08d.pth' % (start_epoch, start_iter)
        enc = HyperNetCoordAE.Encoder(checkpoint_path).cuda()
    else:
        enc = HyperNetCoordAE.Encoder().cuda()

    # decoder
    dec = HyperNetCoordAE.DecoderCenter().cuda()

    if start_iter > 0 or start_epoch > 0:
        print('load from checkpoint: %08d' % start_iter)
        checkpoint_path = '../weights/hyper/dec_%04d_%08d.pth' % (start_epoch, start_iter)
        dec.load_state_dict(torch.load(checkpoint_path))

    # save path
    if not os.path.isdir('../weights/hyper/'):
        os.mkdir('../weights/hyper/')
    if not os.path.isdir('../img_results/hyper/'):
        os.mkdir('../img_results/hyper/')

    # optimizers and loss
    optim_enc = torch.optim.Adam(enc.parameters(), lr=1e-5)
    optim_dec = torch.optim.Adam(dec.parameters(), lr=1e-5)
    loss_func = torch.nn.MSELoss()

    for epoch in range(start_epoch, 10):
        print('epoch: %04d' % epoch)
        loss_avg = 0
        data_tq = tqdm(train_loader)
        for i, data in enumerate(data_tq):
            # data to cuda
            img, coord, img_coord = data

            img = img.cuda()
            coord = coord.cuda()
            img_coord = img_coord.cuda()

            # encoder-decoder
            code = enc.forward(img, grad=True)

            img_recover = dec(code, coord)

            # loss and train
            loss = loss_func(img_recover, img_coord)

            optim_enc.zero_grad()
            optim_dec.zero_grad()
            loss.backward()
            optim_dec.step()
            optim_enc.step()

            with torch.no_grad():
                for param in dec.hyper_net.mapping_layers.parameters():
                    param.data *= 0.9999 + torch.randn_like(param.data) * 0.00005

            # record
            total_iter = start_iter + i + 1
            loss = loss.detach().cpu().numpy()
            loss_avg += loss
            data_tq.set_description('iter: %d loss: %f' % (total_iter, loss))
            # save image
            if total_iter % 100 == 0:
                print('iter: %d loss: %f' % (total_iter, loss_avg/100))
                loss_avg = 0
                d_img = torch.cat([img[0], img[1]], dim=1)
                with torch.no_grad():
                    recover = dec.decode_full(code[0:2])
                i_img = torch.cat([recover[0], recover[1]], dim=1)
                all_img = torch.cat([d_img, i_img], dim=2)
                save_image(all_img, '../img_results/hyper/img_%04d_%08d.png' % (epoch, total_iter))
            # save weights
            if total_iter % 10000 == 0:
                torch.save(dec.state_dict(), '../weights/hyper/dec_%04d_%08d.pth' % (epoch, total_iter))
                enc.save('../weights/hyper/enc_%04d_%08d.pth' % (epoch, total_iter))

        torch.save(dec.state_dict(), '../weights/hyper/dec_%04d_%08d.pth' % (epoch+1, 0))
        enc.save('../weights/hyper/enc_%04d_%08d.pth' % (epoch+1, 0))
        start_iter = 0


if __name__ == '__main__':
    DATA_ROOT = 'D:\\Datasets\\OpenImages'
    START_EPOCH = 1
    START_ITER = 0

    all_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = OpenImages.Train_Coord(data_root=DATA_ROOT, group=0, transform=all_trans)
    data_loader = DataLoader(dataset, batch_size=4, num_workers=12, shuffle=True)

    train(data_loader, START_EPOCH, START_ITER, pretrained_enc=True)
