import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import net
from sampler import InfiniteSamplerWrapper
import torch.nn.functional as F

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.cuda.empty_cache()  # 在合适的地方调用

def train_transform(resize,crop_size):
    transform_list = [
        transforms.Resize(size=(resize,resize)),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def mask_transform(resize,crop_size):
    transform_list = [
        transforms.Resize(size=(resize,resize)),
        transforms.RandomCrop(crop_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

class MaskFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(MaskFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('1')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'MaskFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--fore_dir', type=str, required=True,
                    help='Directory path to a batch of fore images')
parser.add_argument('--mask_dir', type=str, required=True,
                    help='Directory path to a batch of mask images')
parser.add_argument('--back_dir', type=str, required=True,
                    help='Directory path to a batch of back images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=80000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--im_weight', type=float, default=12000.0)
parser.add_argument('--bp_weight', type=float, default=400)
parser.add_argument('--emb_weight', type=float, default=8000.0)
parser.add_argument('--rem_weight', type=float, default=100.0)
parser.add_argument('--tv_weight', type=float, default=0.05)
parser.add_argument('--adv_weight', type=float, default=0.01)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda:0')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder).to(device) 

# 添加鉴别器
discriminator = net.Discriminator().to(device) 

network.train()
discriminator.train()

re_size = 128
fore_tf = train_transform(re_size,re_size) #resize crop
back_tf = train_transform(2*re_size,re_size)
mask_tf = mask_transform(re_size,re_size)

fore_dataset = FlatFolderDataset(args.fore_dir, fore_tf)
back_dataset = FlatFolderDataset(args.back_dir, back_tf)
mask_dataset = MaskFolderDataset(args.mask_dir, mask_tf)

fore_iter = iter(data.DataLoader(
    fore_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(fore_dataset),
    num_workers=args.n_threads))

back_iter = iter(data.DataLoader(
    back_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(back_dataset),
    num_workers=args.n_threads))

mask_iter = iter(data.DataLoader(
    mask_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(mask_dataset),
    num_workers=args.n_threads))

# 将 decoder、PSF、NFFM 和 FEFM 的参数都加入到优化器中
optimizer = torch.optim.Adam(
    list(network.decoder.parameters()) + 
    list(network.PSF.parameters()) + 
    list(network.nffm.parameters()) + 
    list(network.fefm.parameters()),
    lr=args.lr
)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

criterion = nn.BCEWithLogitsLoss()

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    adjust_learning_rate(optimizer_d, iteration_count=i)

    fore_images = next(fore_iter).to(device)
    back_images = next(back_iter).to(device)
    mask_images = next(mask_iter).to(device)

    # 生成器前向传播
    loss_im, loss_bp, loss_rem, loss_tv = network(fore_images, back_images, mask_images)
    loss_im = args.im_weight * loss_im
    loss_bp = args.bp_weight * loss_bp
    loss_rem = args.rem_weight * loss_rem
    loss_tv = args.tv_weight * loss_tv

    # 生成图像（使用 Net 网络的完整前向传播逻辑）
        # 使用 Net 网络生成图像
    back_feats = network.encode_with_intermediate(back_images)
    front_feat = network.encode(fore_images)
    front_feats = network.encode_with_intermediate(fore_images)
    mask_down = F.interpolate(mask_images, scale_factor=1 / 8, mode="nearest")
        
        #剪裁图片
    back_r = F.interpolate(back_images, size=mask_images.shape[-2:], mode='bilinear', align_corners=False)
    mask_E = mask_images.repeat(1, fore_images.shape[1], 1, 1) 
    front_E = back_r * (1 - mask_E) + fore_images * mask_E

        # 提取特征
    feat1 = network.enc_1(fore_images)
    feat2 = network.enc_2(feat1)
    feat3 = network.enc_3(feat2)
    feat4 = network.enc_4(feat3)
    feat_E = network.enc_1(front_E)
    feat_E = network.enc_2(feat_E)
        
    featE = network.fefm(feat_E,feat4)

        # 使用 NFFM 融合特征
    feat_mid1 = network.nffm(feat4, feat3)
    feat_mid2 = network.nffm(feat_mid1, feat2)
    feat_mid3 = network.nffm(feat_mid2, feat1)
    feat_lost = network.nffm(feat_mid3, featE)

        # 使用 PSF 模块生成最终特征
    t = network.PSF(feat_lost, back_feats[-1], mask_down)
    generated_images = network.decoder(t)
        
    # 对抗损失
    real_labels = torch.ones_like(discriminator(generated_images))
    loss_g_adv = criterion(discriminator(generated_images), real_labels) * args.adv_weight
    
    # 合并总损失
    total_loss = loss_im + loss_bp + loss_rem + loss_tv + loss_g_adv

    # 单次生成器更新
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # ================== 鉴别器更新 ==================
    # 真实图像判断
    real_output = discriminator(back_images.detach())  # 注意detach
    loss_d_real = criterion(real_output, torch.ones_like(real_output))
    
    # 生成图像判断
    fake_output = discriminator(generated_images.detach())  # 注意detach
    loss_d_fake = criterion(fake_output, torch.zeros_like(fake_output))
    loss_d = (loss_d_real + loss_d_fake)/2
    # 鉴别器更新
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()
    if (i + 1) % 10 == 0:
        print('im', loss_im)
        print('bp', loss_bp)
        print('r', loss_rem)
        print('t', loss_tv)
        print('d', loss_d)

    writer.add_scalar('loss_im', loss_im.item(), i + 1)
    writer.add_scalar('loss_bp', loss_bp.item(), i + 1)
    writer.add_scalar('loss_total', total_loss.item(), i + 1)
    writer.add_scalar('loss_d', loss_d.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir / 'decoder_iter_{:d}.pth'.format(i + 1))

        state_dict = network.PSF.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir / 'PSF_iter_{:d}.pth'.format(i + 1))
        
        state_dict = network.nffm.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir / 'NFFM_iter_{:d}.pth'.format(i + 1))
        
        state_dict = network.fefm.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir / 'FEFM_iter_{:d}.pth'.format(i + 1))

        state_dict = discriminator.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir / 'discriminator_iter_{:d}.pth'.format(i + 1))

writer.close()