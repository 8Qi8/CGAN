import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import net
from params_position import example_all
import torch.nn.functional as F

device = torch.device('cuda:0')

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

#Camoufalge function
def camouflage(vgg, decoder, PSF, fore, back, mask, NFFM, FEFM, networks):
    b,c,w,h = fore.size()
    down_sam = nn.MaxPool2d((8, 8), (8, 8), (0, 0), ceil_mode=True)
    fore_f = vgg(fore)
    back_f = vgg(back)
    
   #剪裁图片
    back_r = F.interpolate(back, size=(mask.size(2), mask.size(3)), mode='bilinear', align_corners=False)
    mask_E = mask.repeat(1, fore.shape[1], 1, 1) 
    front_E = back_r * (1 - mask_E) + fore * mask_E
    
    feat1 = networks.enc_1(fore)
    feat2 = networks.enc_2(feat1)
    feat3 = networks.enc_3(feat2)
    feat4 = networks.enc_4(feat3)
    
    feat_E = networks.enc_1(front_E)
    feat_E = networks.enc_2(feat_E)
    
    featE = FEFM(feat_E,feat4)
    
    # 使用NFFM融合特征3和特征4
    feat_mid1 = NFFM(feat4, feat3)
    feat_mid2 = NFFM(feat_mid1, feat2)
    feat_mid3 = NFFM(feat_mid2, feat1)
    feat_lost = NFFM(feat_mid3,featE)
    
    mask = down_sam(mask)
    feat = PSF(feat_lost,back_f,mask)
    output = decoder(feat)
    output = output[:,:,:w,:h]
    return output

def embed(fore,mask,back,x,y):
    n_b, c_b, w_b, h_b = back.size()
    n_f, c_f, w_f, h_f = fore.size()

    mask_b = torch.zeros([n_b, 1, w_b, h_b]).to(device)
    fore_b = torch.zeros([n_b, c_b, w_b, h_b]).to(device)

    mask_b[:,:,x:w_f + x, y : h_f+y] = mask
    fore_b[:,:, x:w_f+x, y : h_f+y] = fore
    out = torch.mul(back, 1-mask_b)
    output = torch.mul(fore_b, mask_b) + out
    return output

# Output the coordinates of the upper left corner of the camouflage region,
# the default camouflage region is in the center of the background image.
def position(fore, back):
    a_s, b_s, c_s, d_s = back.size()
    a_c, b_c, c_c, d_c = fore.size()
    x = abs((c_s - c_c) // 2)
    y = abs((d_s - d_c) // 2)
    return x,y

parser = argparse.ArgumentParser()
parser.add_argument('--use_examples', type=int, default=2, help='Use the input and positional parameters we provide. None means input by the users.')
# If input by users
parser.add_argument('--fore', type=str, default='input/fore/2.jpg', help='Foreground image.')
parser.add_argument('--mask', type=str, default='input/mask/2.png', help='Mask image.')
parser.add_argument('--back', type=str, default='input/back/2.jpg', help='Background image.')
parser.add_argument('--zoomSize', type=int, default=1.5, help='Zoom size.')
parser.add_argument('--Vertical', type=int, default=100, help='Move the camouflage region in the vertical direction, the larger the value, the lower the region.')
parser.add_argument('--Horizontal', type=int, default=0, help='Move the camouflage region in the horizontal direction, the larger the value, the more right the region.')
# Crop parameters
parser.add_argument('--Left', type=int, default=0)
parser.add_argument('--Right', type=int, default=-1)
parser.add_argument('--Top', type=int, default=0)
parser.add_argument('--Bottom', type=int, default=-1)

parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='experiments/decoder_iter_80000.pth')
parser.add_argument('--PSF', type=str, default='experiments/PSF_iter_80000.pth')
parser.add_argument('--NFFM', type=str, default='experiments/NFFM_iter_80000.pth')
parser.add_argument('--FEFM', type=str, default='experiments/FEFM_iter_80000.pth')



# Additional options
parser.add_argument('--fore_size', type=int, default=0,
                    help='New (minimum) size for the fore image, \
                    keeping the original size if set to 0')
parser.add_argument('--back_size', type=int, default=0,
                    help='New (minimum) size for the back image, \
                    keeping the original size if set to 0')
parser.add_argument('--mask_size', type=int, default=0,
                    help='New (minimum) size for the mask image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

if args.use_examples:
    assert (args.use_examples>0 and args.use_examples<7)
    example = example_all[args.use_examples-1]
    fore_path = [Path(example['fore_path'])]
    mask_path = [Path(example['mask_path'])]
    back_path = [Path(example['back_path'])]

    zoomSize = example['zoomSize']
    Vertical = example['Vertical']
    Horizontal = example['Horizontal']
    # Crop
    Left = example['Left']
    Right = example['Right']
    Top = example['Top']
    Bottom = example['Bottom']
else:
    assert (args.fore)
    fore_path = [Path(args.fore)]
    assert (args.mask)
    mask_path = [Path(args.mask)]
    assert (args.back)
    back_path = [Path(args.back)]

    zoomSize = args.zoomSize
    Vertical = args.Vertical
    Horizontal = args.Horizontal
    Left = args.Left
    Right = args.Right + 1
    Top = args.Top
    Bottom = args.Bottom + 1




decoder = net.decoder
vgg = net.vgg
PSF = net.PSF(in_planes = 512)
nffm_instance = net.NFFM(512)
fefm_e = net.FEFM(128).to(device)

decoder.eval()
vgg.eval()
PSF.eval()
nffm_instance.eval()
fefm_e.eval()

decoder.load_state_dict(torch.load(args.decoder))
PSF.load_state_dict(torch.load(args.PSF))
nffm_instance.load_state_dict(torch.load(args.NFFM))
# 定义模拟输入
dummy_feat2 = torch.randn(1, 128, 128, 128).to(device)  # 模拟 feat2 的输入
dummy_feat5 = torch.randn(1, 512, 16, 16).to(device)    # 模拟 feat4 的输入

# 通过一次前向传播动态创建 conv1_f2 和 conv1_f5
_ = fefm_e(dummy_feat2, dummy_feat5)

# 加载 FEFM 的权重
fefm_e.load_state_dict(torch.load(args.FEFM))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

networkes = net.Net(vgg, decoder)

vgg.to(device)
decoder.to(device)
PSF.to(device)
nffm_instance.to(device)
networkes.to(device)

fore_tf = test_transform(args.fore_size, args.crop)
back_tf = test_transform(args.back_size, args.crop)
mask_tf = test_transform(args.mask_size, args.crop)

for (fore_path,mask_path) in zip(fore_path, mask_path):
    for back_path in back_path:

        fore = Image.open(str(fore_path))
        back = Image.open(str(back_path))

        # If the foreground is larger than the background, scale the foreground to the background size.
        tempSize = [fore.size[0] * zoomSize, fore.size[1] * zoomSize]
        if tempSize[0] > back.size[0]:
            tempSize[0] = back.size[0]
            tempSize[1] = int(tempSize[1] * back.size[0] / (fore.size[0] * zoomSize))
        if tempSize[1] > back.size[1]:
            temp = tempSize[1]
            tempSize[1] = back.size[1]
            tempSize[0] = int(tempSize[0] * back.size[1] / (temp))

        fore_tf = test_transform((int(tempSize[1]), int(tempSize[0])), args.crop)
        mask_tf = test_transform((int(tempSize[1]), int(tempSize[0])), args.crop)

        fore = fore_tf(fore)
        back = back_tf(back)
        mask = mask_tf(Image.open(str(mask_path)))
        print("fore shape:", fore.shape)
        print("back shape:", back.shape)
        print("mask shape:", mask.shape)

        back = back.to(device).unsqueeze(0)
        fore = fore.to(device).unsqueeze(0)
        mask = mask.to(device).unsqueeze(0)

        mask = (mask > 0).float()
        _, _, w, h = mask.shape

        x, y = position(fore, back)
        Vertical = Vertical if Vertical <= x else x
        Horizontal = Horizontal if Horizontal <= y else y
        x = x + Vertical
        y = y + Horizontal

        back_use = back[:, :, x:x + w, y:y + h]

        with torch.no_grad():
            foreground = fore * mask  # 应用掩码
            background = back_use * (1 - mask)  # 背景区域
            
            result = background+foreground
            result = embed(result, mask, back, x, y)[:, :, Top:Bottom, Left:Right]
            mask_out = torch.zeros([back.size(0), 1,back.size(2), back.size(3)]).to(device)
            mask_out[:,:,x:fore.size(2) + x, y : fore.size(3)+y] = mask
            output_mask = output_dir /'mask'/ '{:s}_{:s}{:s}'.format(back_path.stem, fore_path.stem, args.save_ext)
            save_image(mask_out, str(output_mask))
            output_put = output_dir /'original'/ '{:s}_{:s}{:s}'.format(back_path.stem, fore_path.stem, args.save_ext)
            save_image(result, str(output_put))
            
            output_pre = camouflage(vgg, decoder, PSF, fore, back_use, mask, nffm_instance, fefm_e, networkes)
            output_pre = embed(output_pre, mask, back, x, y)[:, :, Top:Bottom, Left:Right]

        # 根据运行次数调整输出文件名
        output_name = output_dir /'images'/ '{:s}_{:s}{:s}'.format(back_path.stem, fore_path.stem, args.save_ext)
        save_image(output_pre, str(output_name))
        
        # 生成一个与背景大小相同的全黑图片并在其上应用掩码
        

        print(f"Completed run for {back_path.stem}_{fore_path.stem}")