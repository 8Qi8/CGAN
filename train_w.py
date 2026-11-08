import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform(resize, crop_size):
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor()
    ])


def mask_transform(resize, crop_size):
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(crop_size),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        self.paths = list(Path(root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(str(self.paths[index])).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.paths)


class MaskFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        self.paths = list(Path(root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(str(self.paths[index])).convert('1')
        return self.transform(img)

    def __len__(self):
        return len(self.paths)


def adjust_learning_rate(optimizer, iteration_count, args):
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fore_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--back_dir', type=str, required=True)
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

    parser.add_argument('--save_dir', default='./experiments')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=80000)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--im_weight', type=float, default=12000.0)
    parser.add_argument('--bp_weight', type=float, default=400)
    parser.add_argument('--rem_weight', type=float, default=100.0)
    parser.add_argument('--tv_weight', type=float, default=0.05)
    parser.add_argument('--adv_weight', type=float, default=0.01)

    # ⚠️ Windows默认0，后面你可以改成2
    parser.add_argument('--n_threads', type=int, default=0)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(log_dir=str(args.log_dir))

    # ===== 模型 =====
    decoder = net.decoder
    vgg = net.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    network = net.Net(vgg, decoder).to(device)
    discriminator = net.Discriminator().to(device)

    network.train()
    discriminator.train()

    # ===== 数据 =====
    re_size = 128
    fore_dataset = FlatFolderDataset(args.fore_dir, train_transform(re_size, re_size))
    back_dataset = FlatFolderDataset(args.back_dir, train_transform(2 * re_size, re_size))
    mask_dataset = MaskFolderDataset(args.mask_dir, mask_transform(re_size, re_size))

    fore_iter = iter(data.DataLoader(
        fore_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(fore_dataset),
        num_workers=args.n_threads
    ))

    back_iter = iter(data.DataLoader(
        back_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(back_dataset),
        num_workers=args.n_threads
    ))

    mask_iter = iter(data.DataLoader(
        mask_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(mask_dataset),
        num_workers=args.n_threads
    ))

    # ===== 优化器 =====
    optimizer = torch.optim.Adam(
        list(network.decoder.parameters()) +
        list(network.PSF.parameters()) +
        list(network.nffm.parameters()) +
        list(network.fefm.parameters()),
        lr=args.lr
    )

    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # ===== 训练 =====
    for i in tqdm(range(args.max_iter)):

        adjust_learning_rate(optimizer, i, args)
        adjust_learning_rate(optimizer_d, i, args)

        fore = next(fore_iter).to(device)
        back = next(back_iter).to(device)
        mask = next(mask_iter).to(device)

        loss_im, loss_bp, loss_rem, loss_tv = network(fore, back, mask)

        loss_im *= args.im_weight
        loss_bp *= args.bp_weight
        loss_rem *= args.rem_weight
        loss_tv *= args.tv_weight

        # ===== 生成图像 =====
        back_feats = network.encode_with_intermediate(back)
        front_feat = network.encode(fore)

        mask_down = F.interpolate(mask, scale_factor=1/8, mode="nearest")

        back_r = F.interpolate(back, size=mask.shape[-2:], mode='bilinear')
        mask_E = mask.repeat(1, fore.shape[1], 1, 1)
        front_E = back_r * (1 - mask_E) + fore * mask_E

        feat1 = network.enc_1(fore)
        feat2 = network.enc_2(feat1)
        feat3 = network.enc_3(feat2)
        feat4 = network.enc_4(feat3)

        featE = network.fefm(network.enc_2(network.enc_1(front_E)), feat4)

        feat_mid = network.nffm(feat4, feat3)
        feat_mid = network.nffm(feat_mid, feat2)
        feat_mid = network.nffm(feat_mid, feat1)
        feat_lost = network.nffm(feat_mid, featE)

        t = network.PSF(feat_lost, back_feats[-1], mask_down)
        fake = network.decoder(t)

        # ===== GAN =====
        real_label = torch.ones_like(discriminator(fake))
        loss_g_adv = criterion(discriminator(fake), real_label) * args.adv_weight

        total_loss = loss_im + loss_bp + loss_rem + loss_tv + loss_g_adv

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 判别器
        real_out = discriminator(back.detach())
        fake_out = discriminator(fake.detach())

        loss_d = (criterion(real_out, torch.ones_like(real_out)) +
                  criterion(fake_out, torch.zeros_like(fake_out))) / 2

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        if (i + 1) % 10 == 0:
            print(f"[{i}] loss: {total_loss.item():.4f}, D: {loss_d.item():.4f}")

        writer.add_scalar('loss_total', total_loss.item(), i)

    writer.close()


if __name__ == '__main__':
    main()
