import torch
from torch import nn
from models.alpmodule import MultiProtoAsConv
from utils.initialize_weight import initialize_weights
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
from info_nce import InfoNCE

def infonce_pytorch(q_samples, p_samples, n_samples, temp=0.1):

    negative_mode = 'paired' if n_samples.ndim == 3 else 'unpaired'
    infonce = InfoNCE(temperature=temp, negative_mode=negative_mode)
    loss = infonce(q_samples, p_samples, n_samples)
    return loss

# sSE
class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, other_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        # output_tensor = torch.mul(input_tensor, squeeze_tensor)
        output_tensor = torch.mul(other_tensor, squeeze_tensor) # Spatially Recalibrate
        return output_tensor

class conv_ori_ch(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_ori_ch, self).__init__()
        self.conv_ori = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv_ori(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block_sup(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_sup, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# options for type of prototypes
FG_PROT_MODE = 'gridconv+'
BG_PROT_MODE = 'mask'

# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95

mse_loss = torch.nn.MSELoss(reduction='mean')


def get_masks(sup_lb, class_id):
    class_ids = [1]

    fg_mask = torch.where(sup_lb == class_id,
                          torch.ones_like(sup_lb), torch.zeros_like(sup_lb))
    bg_mask = torch.where(sup_lb != class_id,
                          torch.ones_like(sup_lb), torch.zeros_like(sup_lb))
    for class_id in class_ids:
        bg_mask[sup_lb == class_id] = 0
    # fg_numpy = fg_mask.cpu().detach().numpy()
    # bg_numpy = bg_mask.cpu().detach().numpy()

    return fg_mask.float(), bg_mask.float()

class AAS_DCL(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(AAS_DCL, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=32)
        self.Conv1_ = conv_block(ch_in=4, ch_out=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)
        self.Conv5 = conv_ori_ch(ch_in=256, ch_out=512)  # bottleneck

        # self.sSE = SpatialSELayer(num_channels=16)
        self.sSE1 = SpatialSELayer(num_channels=32)
        self.sSE2 = SpatialSELayer(num_channels=64)
        self.sSE3 = SpatialSELayer(num_channels=128)
        self.sSE4 = SpatialSELayer(num_channels=256)
        self.sSE5 = SpatialSELayer(num_channels=512)

        self.alp_unit4 = MultiProtoAsConv(proto_grid=[4, 4],
                                          feature_hw=[16, 16])
        self.alp_unit_final = MultiProtoAsConv(proto_grid=[64, 64],
                                          feature_hw=[256, 256])

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.conv_fusion = conv_ori_ch(ch_in=512+513, ch_out=512)
        self.getCorrConv = conv_ori_ch(ch_in=32*2, ch_out=32)
        self.conv_fusion_ = conv_ori_ch(ch_in=32 + 33, ch_out=32)

        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True), # 2.3 ori 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0))

        self.AdaptiveAvg = nn.AdaptiveAvgPool2d((256,256))
        self.GAP = nn.AdaptiveAvgPool2d(1)

        kernel_size = [ft_l // grid_l for ft_l, grid_l in zip([16, 16], [4, 4])]
        self.avg_pool_op = nn.AvgPool2d(kernel_size)

        initialize_weights(self)

    def global_pro(self, s_in, fg_mask, bg_mask):
        _1, _2, h, w = s_in.size()
        s_lb = F.interpolate(fg_mask, size=[h, w], mode="bilinear")
        s_lb2 = F.interpolate(bg_mask, size=[h, w], mode="bilinear")
        vec_pos = torch.sum(s_in * s_lb, dim=(-1, -2)) \
                  / (s_lb.sum(dim=(-1, -2)) + 1e-5)
        vec_pos2 = torch.sum(s_in * s_lb2, dim=(-1, -2)) \
                   / (s_lb2.sum(dim=(-1, -2)) + 1e-5)
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
        vec_pos2 = vec_pos2.unsqueeze(dim=2).unsqueeze(dim=3)
        vec = torch.cat((vec_pos, vec_pos2), dim=1)
        pro = vec_pos + vec_pos2
        return vec, pro

    def SpatialContext(self, x):
        x = self.sSE5(x, x)
        x = self.avg_pool_op(x)
        x = x.reshape(-1, 1, 512)
        return x

    def Consin(self, s_in, q_in, fg_mask, bg_mask):
        vec, _ = self.global_pro(s_in, fg_mask, bg_mask)
        sim = F.cosine_similarity(torch.cat((q_in, q_in), dim=1), vec, dim=1, eps=1e-7)
        x_sq = q_in * sim
        return sim

    def SimPrior(self, s_in, q_in, fg_mask, bg_mask):
        sup_lb_ = F.interpolate(fg_mask.float(), size=[q_in.shape[-2], q_in.shape[-1]], mode="nearest")
        similarity = F.cosine_similarity(s_in * sup_lb_, q_in, dim=1, eps=1e-7)
        sim_nor = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + 1e-7)
        _, pro = self.global_pro(s_in, fg_mask, bg_mask)
        pro = pro.expand(-1, -1, sim_nor.shape[-2], sim_nor.shape[-1])
        return torch.cat((sim_nor.unsqueeze(0), pro), dim=1)

    def getCCL(self, x_q, x_s, x_nons): # CCL
        x_q = self.SpatialContext(x_q)
        x_s = self.SpatialContext(x_s)

        length = x_q.shape[0]
        ccl = 0.
        x_nons_ = []
        for i in range(length):
            x_q_ = x_q[i]
            x_s_ = x_s[i]
            for x_non in x_nons:
                x_non = self.SpatialContext(x_non)
                ccl += infonce_pytorch(x_q_, x_s_, x_non.view(x_non.shape[0], -1), temp=0.1)
        return ccl / (len(x_nons) * length)

    def getPCL(self, x_q, x_s, fg_mask, bg_mask, x_nons, fg_m_nons, bg_m_nons): # CPCL

        _, sup_pro = self.global_pro(x_s, fg_mask, bg_mask)  # support global pro

        non_pros = []
        for i in range(len(fg_m_nons)):
            _, non_pro = self.global_pro(x_nons[i], fg_m_nons[i], bg_m_nons[i])  #
            non_pros.append(non_pro)
        pcl_loss = infonce_pytorch(self.GAP(x_q).view(1, -1), sup_pro.view(1, -1),
                                   (torch.stack(non_pros, dim=1)).view(len(fg_m_nons), -1),
                                   temp=0.1)
        return pcl_loss

    def getPCL2(self, x_q, x_s, sup_lb, x_nons, lb_nons):  # PPCL

        sup_lb_down = F.interpolate(sup_lb, size=[x_s.shape[-1], x_s.shape[-1]], mode="nearest")
        sup_pro_n, sup_pro_n_b, que_grid_sim = self.alp_unit4(x_q, x_s, sup_lb_down, mode='gridconv', thresh=FG_THRESH,
                                                              isval=False, val_wsize=None, vis_sim=False)
        pcl_loss = 0.
        cnt = 0
        x_q_mean = self.GAP(x_q).view(1, -1)
        for i in range(len(lb_nons)):
            lb_non_down = F.interpolate(lb_nons[i], size=[x_nons[i].shape[-1], x_nons[i].shape[-1]], mode="nearest")
            non_pro_n, non_pro_n_b, aux_attr = self.alp_unit4(x_q, x_nons[i], lb_non_down, mode='gridconv',
                                                              thresh=FG_THRESH, isval=False, val_wsize=None,
                                                              vis_sim=False)
            pcl_l = 0.
            if len(non_pro_n) > 0:
                non_pro_n = non_pro_n + sup_pro_n_b + non_pro_n_b 
                for j in range(len(sup_pro_n)):
                    cnt += 1
                    pcl_l = infonce_pytorch(x_q_mean, sup_pro_n[j].view(1, -1),
                                            (torch.stack(non_pro_n, dim=1)).view(len(non_pro_n), -1),
                                            temp=0.1)

            pcl_loss += pcl_l
        pcl_loss = pcl_loss / (cnt if cnt > 0 else 1)
        return pcl_loss

    def getFtsSim(self, s_fts, q_fts, sup_lb):
        sim_fts = []
        for i in range(len(s_fts)):
            sup_lb_ = F.interpolate(sup_lb.float(), size=[s_fts[i].shape[-2], s_fts[i].shape[-1]], mode="nearest")
            sim_layer = F.cosine_similarity(s_fts[i], q_fts[i], dim=1, eps=1e-7)
            # sim_nor = (sim_layer - sim_layer.min(1)[0].unsqueeze(1)) / (
            #             sim_layer.max(1)[0].unsqueeze(1) - sim_layer.min(1)[0].unsqueeze(1) + 1e-7)
            sim_layer_nor = self.AdaptiveAvg(sim_layer)
            sim_fts.append(sim_layer_nor)
        sim_fts = torch.mean(torch.stack(sim_fts, dim=1), dim=1)
        # sim_fts = F.normalize(sim_fts, dim=1)
        sim_fts = (sim_fts - sim_fts.min(1)[0].unsqueeze(1)) / (
                    sim_fts.max(1)[0].unsqueeze(1) - sim_fts.min(1)[0].unsqueeze(1) + 1e-7)
        return sim_fts

    def forward(self, sup_img, que_img, sup_lb, que_init, class_id, img_nons, super_masks, super_ids, query_name):

        class_id = 1
        fg_mask, bg_mask = get_masks(sup_lb, class_id)

        # encoding path
        x1 = self.Conv1_(torch.cat((sup_img, sup_lb), dim=1))
        x1_q = self.Conv1(que_img)

        x2 = self.Maxpool(x1)
        x2_q = self.Maxpool(x1_q)
        x2_q = self.sSE1(x2, x2_q)
        x2 = self.Conv2(x2)
        x2_q = self.Conv2(x2_q)

        x3 = self.Maxpool(x2)
        x3_q = self.Maxpool(x2_q)
        x3_q = self.sSE2(x3, x3_q)
        x3 = self.Conv3(x3)
        x3_q = self.Conv3(x3_q)

        x4 = self.Maxpool(x3)
        x4_q = self.Maxpool(x3_q)
        x4_q = self.sSE3(x4, x4_q)
        x4 = self.Conv4(x4)
        x4_q = self.Conv4(x4_q)

        x5 = self.Maxpool(x4)
        x5_q = self.Maxpool(x4_q)
        x5_q = self.sSE4(x5, x5_q)
        x5 = self.Conv5(x5)  # bottleneck
        x5_q = self.Conv5(x5_q)  # 1,512,16,16
        x5_q = self.sSE5(x5, x5_q)
        # encoder end--------------

        dcl_loss = 0.
        x_nons = []
        fg_m_nons = []
        bg_m_nons = []

        for i in range(len(super_masks)):
            fg_m, bg_m = get_masks(super_masks[i], super_ids[i])
            fg_m_nons.append(fg_m)
            bg_m_nons.append(bg_m)

        # with torch.no_grad():
        for i in range(len(img_nons)):
            x1_non = self.Conv1(img_nons[i])
            x2_non = self.Maxpool(x1_non)
            x2_non = self.Conv2(x2_non)
            x3_non = self.Maxpool(x2_non)
            x3_non = self.Conv3(x3_non)
            x4_non = self.Maxpool(x3_non)
            x4_non = self.Conv4(x4_non)
            x5_non = self.Maxpool(x4_non)
            x5_non = self.Conv5(x5_non)
            x_nons.append(x5_non)

        sim_prior5 = self.SimPrior(x5, x5_q, fg_mask, bg_mask)
        x5_sq = self.conv_fusion(torch.cat((x5_q, sim_prior5), dim=1))

        # # ------- Prototypical CL --------
        loss1 = self.getPCL(x5_sq, x5, fg_mask, bg_mask, x_nons, fg_m_nons, bg_m_nons) # CCPL
        loss2 = self.getPCL2(x5_sq, x5, sup_lb, x_nons, super_masks) # DCPL

        dcl_loss = loss1 * 0.4 + loss2 * 0.6

        # # ------- Contextual CL --------
        loss3 = self.getCCL(x5_sq, x5, x_nons)
        dcl_loss += (loss3 * 0.3)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5_q = self.Up5(x5_q)
        d5 = torch.cat((x4, d5), dim=1)
        d5_q = torch.cat((x4_q, d5_q), dim=1)
        d5 = self.Up_conv5(d5)
        d5_q = self.Up_conv5(d5_q)
        d5_q = self.sSE4(d5, d5_q)

        d4 = self.Up4(d5)
        d4_q = self.Up4(d5_q)
        d4 = torch.cat((x3, d4), dim=1)
        d4_q = torch.cat((x3_q, d4_q), dim=1)
        d4 = self.Up_conv4(d4)
        d4_q = self.Up_conv4(d4_q)
        d4_q = self.sSE3(d4, d4_q)

        d3 = self.Up3(d4)
        d3_q = self.Up3(d4_q)
        d3 = torch.cat((x2, d3), dim=1)
        d3_q = torch.cat((x2_q, d3_q), dim=1)
        d3 = self.Up_conv3(d3)
        d3_q = self.Up_conv3(d3_q)
        d3_q = self.sSE2(d3, d3_q)

        d2 = self.Up2(d3)
        d2_q = self.Up2(d3_q)
        d2 = torch.cat((x1, d2), dim=1)
        d2_q = torch.cat((x1_q, d2_q), dim=1)
        d2 = self.Up_conv2(d2)
        d2_q = self.Up_conv2(d2_q)
        d2_q = self.sSE1(d2, d2_q)

        d1_q = self.Conv_1x1(d2_q)
        res = F.softmax(d1_q)

        mse = 0.

        sim_mask = F.cosine_similarity(res, torch.cat((bg_mask, fg_mask), dim=1), dim=1, eps=1e-7)  # 1,256,256
        # sim_mask = F.normalize(sim_mask, dim=1)
        sim_mask = (sim_mask - sim_mask.min(1)[0].unsqueeze(1)) / (
                sim_mask.max(1)[0].unsqueeze(1) - sim_mask.min(1)[0].unsqueeze(1) + 1e-7)
        sim_fts = self.getFtsSim([x1, x2, x3, x4, x5], [x1_q, x2_q, x3_q, x4_q, x5_q], fg_mask)
        mse = mse_loss(sim_mask.float(), sim_fts.float())

        I = 5
        low_mse = mse
        que_pre = res
        for i in range(I):

            d2_q_ = self.getCorrConv(torch.cat((d2_q * que_pre[:, 0, :, :], d2_q * que_pre[:, 1, :, :]), dim=1))
            sim_p = self.SimPrior(d2, d2_q_, fg_mask, bg_mask)
            d1_p_ = self.conv_fusion_(torch.cat((d2_q_, sim_p), dim=1))
            d1_q = self.Conv_1x1(d1_p_)
            que_pre = F.softmax(d1_q)


            sim_mask = F.cosine_similarity(que_pre, torch.cat((bg_mask, fg_mask), dim=1), dim=1, eps=1e-7)
            # sim_mask = F.normalize(sim_mask, dim=1)
            sim_mask = (sim_mask - sim_mask.min(1)[0].unsqueeze(1)) / (
                    sim_mask.max(1)[0].unsqueeze(1) - sim_mask.min(1)[0].unsqueeze(1) + 1e-7)
            mse = mse_loss(sim_mask.float(), sim_fts.float())

            if low_mse >= mse:
                res = que_pre
                low_mse = mse

        return res, mse, dcl_loss
