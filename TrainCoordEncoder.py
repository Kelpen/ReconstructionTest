from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm

from torchvision.utils import save_image

from Datasets import OpenImages
from Models import CoordEncoder

from itertools import chain


def train_general(train_loader, start_iter=0):
    enc = CoordEncoder.Encoder().cuda()
    dec = CoordEncoder.MemDecoder().cuda()

    if start_iter > 0:
        print('load from checkpoint: %08d' % start_iter)
        checkpoint_path = 'weights/weight_norm/dec_%08d.pth' % start_iter
        checkpoint = torch.load(checkpoint_path)
        dec.load_state_dict(checkpoint)

    optim_enc = torch.optim.Adam(enc.parameters(), lr=3e-5)
    optim_dec = torch.optim.Adam(dec.parameters(), lr=1e-5)
    loss_func = torch.nn.MSELoss()

    data_tq = tqdm(train_loader)
    for i, data in enumerate(data_tq):
        data = data.cuda()

        if i % 10 == 0:
            enc.train()
            code = enc.forward(data, grad=True)
        else:
            code = enc.forward(data, grad=False)
        img = dec(code)
        loss = loss_func(img, data)

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
            d_img = torch.cat([data[0], data[1]], dim=1)
            i_img = torch.cat([img[0], img[1]], dim=1)
            all_img = torch.cat([d_img, i_img], dim=2)
            save_image(all_img, 'img_results/weight_norm/img_%08d.png' % total_iter)
        if total_iter % 1000 == 0:
            torch.save(dec.state_dict(), 'weights/weight_norm/dec_%08d.pth' % total_iter)


def train_by_trainer(train_loader):
    enc = CoordEncoder.Encoder().cuda()
    dec = CoordEncoder.MemDecoderTrainer(torch.optim.Adam, lr_conv=1e-4, lr_line=1e-4).cuda()
    loss_func = torch.nn.MSELoss()

    data_tq = tqdm(train_loader)
    for i, data in enumerate(data_tq):
        data = data.cuda()
        code = enc(data)
        loss = dec(code, data, loss_func)

        data_tq.set_description('loss: %f' % loss.detach().cpu().numpy())

        if i % 100 == 0:
            with torch.no_grad():
                code = enc(data[:1])
                img = dec.test(code)
                save_image(img[0], 'img_results/coord_sqrt/img_%08d.png' % i)
        if i % 1000 == 0:
            torch.save(dec.state_dict(), 'weights/coord_sqrt/dec_%08d.pth' % i)


if __name__ == '__main__':
    DATA_ROOT = 'F:\\Datasets\\OpenImages'
    pretrained_iter = 62000

    all_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = OpenImages.Train0(data_root=DATA_ROOT, transform=all_trans)
    data_loader = DataLoader(dataset, batch_size=2, num_workers=8, shuffle=True)

    train_general(data_loader, pretrained_iter)
    # train_by_trainer(data_loader)
