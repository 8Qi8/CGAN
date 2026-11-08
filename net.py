import torch.nn as nn
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from function import patch_mv_norm, patch_adain

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(513, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

winsize = 7


class FEFM(nn.Module):
    def __init__(self, out_channels):
        super(FEFM, self).__init__()
        self.out_channels = out_channels

        # 定义通用的降维卷积（输入通道数会在 forward 中动态调整）
        self.conv1_f2 = None  # f2 的 1x1 卷积将在 forward 中动态定义
        self.conv1_f5 = None  # f5 的 1x1 卷积将在 forward 中动态定义

        # 后续固定的卷积
        self.conv3x3 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)  # 3x3 卷积
        self.conv1x1_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # 1x1 卷积
        self.sigmoid = nn.Sigmoid()  # 输出激活函数

    def forward(self, f2, f5):
        # 动态定义 f2 和 f5 的降维卷积
        if self.conv1_f2 is None:
            self.conv1_f2 = nn.Conv2d(f2.size(1), self.out_channels, kernel_size=1).to(f2.device)
        if self.conv1_f5 is None:
            self.conv1_f5 = nn.Conv2d(f5.size(1), self.out_channels, kernel_size=1).to(f5.device)

        # 1x1 卷积降维
        f2_ = self.conv1_f2(f2)  # (B, out_channels, H, W)
        f5_ = self.conv1_f5(f5)  # (B, out_channels, H/8, W/8)

        # 上采样 f5 到 f2 的分辨率
        f5_upsampled = F.interpolate(f5_, size=f2_.shape[2:], mode='bilinear', align_corners=True)  # (B, out_channels, H, W)

        # 拼接高低分辨率特征
        fused = torch.cat([f2_, f5_upsampled], dim=1)  # (B, out_channels * 2, H, W)

        # 3x3 卷积和 1x1 卷积
        fused = self.conv3x3(fused)  # (B, out_channels, H, W)
        fused = self.conv1x1_out(fused)  # (B, out_channels, H, W)

        # 残差连接：加入 f2_ 以增强信息传递
        fused = fused + f2_  # (B, out_channels, H, W)

        # Sigmoid 激活
        out = self.sigmoid(fused)  # (B, out_channels, H, W)

        return out
    
class PSF(nn.Module):
    def __init__(self, in_planes):
        super(PSF, self).__init__()
        self.e = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.residual_conv = nn.Conv2d(in_planes, in_planes, (1, 1))  # 用于调整前景特征维度

    def forward(self, front, back, mask):
        # 局部自适应特征
        EE = self.e(patch_adain(front, back, winsize))

        # 结构特征
        FF = self.f(patch_mv_norm(front, winsize))
        GG = self.g(patch_mv_norm(back, winsize))

        # 外观特征
        HH = self.h(back)

        # 计算相似度
        b, _, h, w = GG.size()
        FF = FF.view(b, -1, w * h)
        GG = GG.view(b, -1, w * h)
        F_n = (FF * FF).sum(dim=1).sqrt()
        G_n = (GG * GG).sum(dim=1).sqrt()
        S = torch.mul(FF, GG).sum(dim=1) / (F_n * G_n)
        S_n = ((S - S.min(dim=1)[0].unsqueeze(1)) /
               (S.max(dim=1)[0].unsqueeze(1) - S.min(dim=1)[0].unsqueeze(1))).view(b, 1, h, w)

        # 融合特征
        fused = torch.mul(S_n, EE) + torch.mul(1 - S_n, HH)

        # 引入残差连接
        residual = self.residual_conv(front)  # 通道数一致后的前景特征
        # 直接进行加法
        fused_with_residual = fused + residual

        # 使用残差连接的特征与 mask 拼接
        O = torch.cat((fused_with_residual, mask), dim=1)

        return O

class WeightedFusion(nn.Module):
    def __init__(self, in_channels):
        super(WeightedFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, S1, S2):
        combined = torch.cat([S1, S2], dim=1)
        alpha = self.mlp(combined)
        fused = alpha * S1 + (1 - alpha) * S2
        return fused

class NFFM(nn.Module):
    def __init__(self, out_channels):
        super(NFFM, self).__init__()
        self.out_channels = out_channels
        self.conv3x3 = nn.Conv2d(in_channels=self.out_channels * 2, out_channels=512, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.weighted_fusion = WeightedFusion(512)

    def forward(self, P1, P2):
        in_channels1 = P1.size(1)
        in_channels2 = P2.size(1)

        conv1 = nn.Conv2d(in_channels1, self.out_channels, kernel_size=1).to(P1.device)
        conv2 = nn.Conv2d(in_channels2, self.out_channels, kernel_size=1).to(P2.device)

        P1 = conv1(P1)
        P2 = conv2(P2)

        if P1.size() != P2.size():
            P2 = F.interpolate(P2, size=P1.size()[2:], mode='bilinear', align_corners=False)

        S1 = P1 * P2
        S2 = torch.cat((P1, P2), dim=1)

        S2 = self.conv3x3(S2)
        S2 = self.bn(S2)
        S2 = self.relu(S2)

        # 使用加权融合替代简单相加
        K = self.weighted_fusion(S1, S2)

        return K

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.nffm = NFFM(512)
        self.fefm = FEFM(128)
        self.mse_loss = nn.MSELoss()
        self.PSF = PSF(in_planes=512)

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def encode_with_intermediate_a(self, input):
        results = [input]
        for i in range(4, 12):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def tv_loss(self, target):
        assert (len(target.size()) == 4)
        dx = (target[:, :, 1:, :-1] - target[:, :, :-1, :-1]) ** 2
        dy = (target[:, :, :-1, 1:] - target[:, :, :-1, :-1]) ** 2
        dxy = dx + dy
        tv = dxy.sum()
        return tv

    def normalize(self, a):
        b = np.zeros(a.shape)
        for k in range(a.shape[0]):
            max = np.max(a[k])
            min = np.min(a[k])
            m = a[k].shape[0]
            n = a[k].shape[1]
            for i in range(m):
                for j in range(n):
                    b[k, i, j] = (a[k, i, j] - min) / (max - min + 1e-5)
        return b

    def attention(self, f):
        n, c, w, h = f.shape
        ff = np.zeros((n, 1, w, h))
        af = np.zeros((n, 1, w, h))
        for i in range(c):
            ff[:, 0, :, :] += f[:, i, :, :]
        ff = ff / c  # n,w,h
        for i in range(f.shape[0]):
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            a, b = saliency.computeSaliency(ff[i, 0])
            af[i, 0] = (b - b.min()) / (b.max() - b.min())
        return af

    def remove_loss(self, input, target, a_n):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        a_n = a_n.view(1, -1)
        d = input.size(1)
        input = input.transpose(0, 1).contiguous().view(d, -1)
        target = target.transpose(0, 1).contiguous().view(d, -1)
        input = torch.mul(input, a_n)
        target = torch.mul(target, a_n)
        return self.mse_loss(input, target)

    def bp_loss(self, x, y, n):
        (b, c, h, w) = x.size()
        zeroPad2d = torch.nn.ZeroPad2d(n // 2)

        x_pad = zeroPad2d(x)
        x_fold = F.unfold(x_pad, (n, n), stride=1).view(b, c, n * n, -1)
        x_mean = x_fold.mean(dim=2)
        x_std = x_fold.var(dim=2).sqrt()

        y_pad = zeroPad2d(y)
        y_fold = F.unfold(y_pad, (n, n), stride=1).view(b, c, n * n, -1)
        y_mean = y_fold.mean(dim=2)
        y_std = y_fold.var(dim=2).sqrt()
        return self.mse_loss(x_mean, y_mean) + \
            self.mse_loss(x_std, y_std)

    def imm_loss(self, input, target, a, mask):
        assert (input.size() == target.size())
        n, c, w, h = input.shape
        maskk = (mask.view(n, 1, 1, w * h) + mask.view(n, 1, w * h, 1)).clip(0, 1)  # 只关注那些都在mask里的点
        input = input.view(n, c, 1, w * h) - input.view(n, c, w * h, 1)
        target = target.view(n, c, 1, w * h) - target.view(n, c, w * h, 1)
        aa = a.view(n, 1, 1, w * h) + a.view(n, 1, w * h, 1)
        return self.mse_loss(input * aa * maskk, target * aa * maskk)

    def forward(self, front, back, mask):
        device = torch.device("cuda:0")
        back_feats = self.encode_with_intermediate(back)
        front_feat = self.encode(front)
        front_feats = self.encode_with_intermediate(front)
        mask_down = F.interpolate(mask, scale_factor=1 / 8, mode="nearest")
        
        #剪裁图片
        back_resized = F.interpolate(back, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        mask_E = mask.repeat(1, front.shape[1], 1, 1) 
        front_E = back_resized * (1 - mask_E) + front * mask_E
        
        # 提取特征
        feat1 = self.enc_1(front)
        feat2 = self.enc_2(feat1)
        feat3 = self.enc_3(feat2)
        feat4 = self.enc_4(feat3)
        
        feat_E = self.enc_1(front_E)
        feat_E = self.enc_2(feat_E)
        
        featE = self.fefm(feat_E,feat4)

        # 使用NFFM融合特征3和特征4
        feat_mid1 = self.nffm(feat4, feat3)
        feat_mid2 = self.nffm(feat_mid1, feat2)
        feat_mid3 = self.nffm(feat_mid2, feat1)
        feat_lost = self.nffm(feat_mid3, featE)

        a = torch.from_numpy(self.attention(front_feat.cpu().numpy())).float().to(device)
        t = self.PSF(feat_lost, back_feats[-1], mask_down)
        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_cam = self.imm_loss(g_t_feats[-1], front_feat, a, mask_down)

        nnn = winsize
        loss_bp = self.bp_loss(g_t_feats[0], back_feats[0], nnn)
        loss_tv = self.tv_loss(g_t)
        a = torch.from_numpy(self.attention(front_feats[0].cpu().numpy())).float().to(device)
        a_n = 1 - a
        remove = self.remove_loss(g_t_feats[0], back_feats[0], a_n)
        for i in range(1, 4):
            loss_bp += self.bp_loss(g_t_feats[i], back_feats[i], nnn)
            a = torch.from_numpy(self.attention(front_feats[i].cpu().numpy())).float().to(device)
            a_n = 1 - a
            remove += self.remove_loss(g_t_feats[i], back_feats[i], a_n)
        return loss_cam, loss_bp, remove, loss_tv

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)