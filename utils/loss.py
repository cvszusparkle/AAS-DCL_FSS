import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

__all__ = [
    "DiceLoss",
    "DiceBCELoss",
    "DiceCELoss",
    "DiceJaccardLoss",
    "DiceLossMulti"
]


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(output, target, smooth=1e-5):
        y_pd = output.view(-1)
        y_gt = target.view(-1)
        intersection = torch.sum(y_pd * y_gt)
        score = (2. * intersection + smooth) / (torch.sum(y_pd) + torch.sum(y_gt) + smooth)
        loss = 1 - score
        # batch = target.size(0)
        # input_flat = output.view(batch, -1)
        # target_flat = target.view(batch, -1)
        #
        # intersection = input_flat * target_flat
        # loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        # loss = 1 - loss.sum() / batch
        return loss


class DiceBCELoss(nn.Module):
    def __init__(self, weight_ce=0.8): # 0.6 ori change 0602
        super(DiceBCELoss, self).__init__()
        self.weight_ce = weight_ce
        self.ce = nn.BCELoss()
        self.dc = DiceLoss()

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        result = self.weight_ce * ce_loss + (1 - self.weight_ce) * dc_loss
        return result

class DiceCELoss(nn.Module):
    def __init__(self, weight_ce=0.8):
        super(DiceCELoss, self).__init__()
        self.weight_ce = weight_ce
        self.ce = nn.CrossEntropyLoss()
        self.dc = DiceLoss()

    def forward(self, net_output, target):
        mask = torch.argmax(target, dim=1)
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, mask.long())
        result = self.weight_ce * ce_loss + (1 - self.weight_ce) * dc_loss
        return result

class DiceJaccardLoss(nn.Module):
    def __init__(self):
        super(DiceJaccardLoss, self).__init__()
        self.ce = nn.BCELoss()

    def forward(self, outputs, targets):
        eps = 1e-15
        jaccard_target = (targets == 1).float()
        jaccard_output = outputs
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        loss = self.ce(outputs, targets) - torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class DiceLossMulti(nn.Module):
    def __init__(self):
        super(DiceLossMulti, self).__init__()

    @staticmethod
    def forward(output, target, smooth=1e-5):
        n_classes = 15
        loss_list = []
        for i in range(n_classes):
            y_pd = output[:, i:i+1, :, :]
            y_gt = target[:, i:i+1, :, :]
            y_pd = y_pd.contiguous().view(-1)
            y_gt = y_gt.contiguous().view(-1)
            intersection = torch.sum(y_pd * y_gt)
            score = (2. * intersection + smooth) / (torch.sum(y_pd) + torch.sum(y_gt) + smooth)
            loss = 1 - score
            loss_list.append(loss)
        loss = sum(loss_list)/len(loss_list)

        return loss

class CriterionMulti(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionMulti, self).__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLossMulti()

    def forward(self, preds, target):
        # 0：predict 1:coarse predict 2: fine predict
        n_classes = 15
        mask = torch.argmax(target, dim=1)
        loss1 = self.ce(preds[0], mask) / n_classes
        loss1_dice = self.dice(preds[0], target)

        loss2 = self.ce(preds[1], mask) / n_classes
        loss2_dice = self.dice(preds[1], target)

        loss3 = self.ce(preds[2], mask) / n_classes
        loss3_dice = self.dice(preds[2], target)

        loss4 = self.ce(preds[3], mask) / n_classes
        loss4_dice = self.dice(preds[3], target)

        # pred = F.sigmoid(preds[0] + preds[2])
        # loss_dice = self.dice(preds[0], target)
        # L = λa · la + λc · lc + λf · lf
        # return 0.7 * loss1 + 0.6 * loss2 + 0.4 * loss3 + loss4
        # loss0
        # return 0.1 * (loss1+loss1_dice) + 0.2 * (loss2+loss2_dice) + 0.3 * (loss3+loss3_dice) + 0.4*(loss4+loss4_dice)
        # loss1
        return (loss1 + loss1_dice) + 0.2 * (loss2 + loss2_dice) + 0.4 * (loss3 + loss3_dice) + 0.6 * (loss4 + loss4_dice)
        #return loss1 + 0.2 * loss2 + 0.4 * loss3 + 0.6 * loss4

class CELossMulti(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CELossMulti, self).__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # 0：predict 1:coarse predict 2: fine predict
        mask = torch.argmax(target, dim=1) # B * C * H * W --> B * H * W
        loss = self.ce(pred, mask)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # def forward(self, output1, output2, label):
    #     euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    #     loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
    #                                   (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
    #     return loss_contrastive
    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2) +
                                      torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive