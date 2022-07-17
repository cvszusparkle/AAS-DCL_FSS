import torch
import torch
from torch import nn
import numpy as np
from sklearn import metrics
# from .thirdparty import sklearn_metrics

__all__ = [
    "get_f1_score",
    "get_jaccard_score",
    "get_accuracy",
    "get_specificity",
    "get_sensitivity",
    "get_f1_score_multi",
    "get_jaccard_score_multi",
    "get_accuracy_multi",
    "get_specificity_multi",
    "get_sensitivity_multi",
]


# https://github.com/pytorch/pytorch/issues/1249 #
def get_f1_score(pred, target):
    smooth = 1.
    pred = (pred >= 0.5).float()
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return float((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth))


def get_f1_score2(pd, gt, threshold=0.5):
    """
    params: pd==prediction
    params: gt: ground truth
    return: dice coefficient / f1-score
    """

    pd = (pd >= threshold).float()
    intersection = torch.sum((pd + gt) == 2)

    score = float(2 * intersection) / (float(torch.sum(pd) + torch.sum(gt)) + 1e-6)
    return score


def get_jaccard_score(pd, gt, threshold=0.5):
    """
    params: pd==prediction
    params: gt: ground truth
    return: jaccard similarity / iou score
    """

    pd = (pd > threshold).float()
    intersection = torch.sum((pd + gt) == 2)
    union = torch.sum((pd + gt) >= 1)

    score = float(intersection) / (float(union) + 1e-6)
    return score




def get_accuracy(pd, gt, threshold=0.5):
    """
    params: pd==prediction
    params: gt: ground truth
    return: accuracy score
    """

    pd = (pd > threshold).float()
    corr = torch.sum(pd == gt)
    # tensor_size = pd.size(0) * pd.size(1) * pd.size(2) * pd.size(3)
    tensor_size = pd.size(0) * pd.size(1) * pd.size(2)

    score = float(corr) / float(tensor_size)
    return score


def get_sensitivity(pd, gt, threshold=0.5):
    """
    params: pd==prediction
    params: gt: ground truth
    return: sensitivity score / recall rate
    """

    pd = (pd > threshold).float()
    tp = (((pd == 1).float() + (gt == 1).float()) == 2).float()  # True Positive
    fn = (((pd == 0).float() + (gt == 1).float()) == 2).float()  # False Negative

    score = float(torch.sum(tp)) / (float(torch.sum(tp + fn)) + 1e-6)
    return score


def get_specificity(pd, gt, threshold=0.5):
    pd = (pd > threshold).float()
    tn = (((pd == 0).float() + (gt == 0).float()) == 2).float()  # True Negative
    fp = (((pd == 1).float() + (gt == 0).float()) == 2).float()  # False Positive

    score = float(torch.sum(tn)) / (float(torch.sum(tn + fp)) + 1e-6)
    return score

#------------------------------------------------------------------------------------------

def get_f1_score_multi(pd, gt, n_classes):
    """
    params: pd==prediction
    params: gt: ground truth
    return: dice coefficient / f1-score
    """
    score = []
    for i in range(n_classes): # 0812
        score.append(get_f1_score(pd[:, i:i+1, :, :], gt[:, i:i+1, :, :]))
        # score.append(get_f1_scorew(pd[:, i:i+1, :, :], gt[:, i:i+1, :, :])) # w师兄的 # 0527 change
    score = sum(score) / n_classes
    return score

def get_jaccard_score_multi(pd, gt, n_classes):
    """
    params: pd==prediction
    params: gt: ground truth
    return: jaccard similarity / iou score
    """

    score = []
    for i in range(n_classes):
        score.append(get_jaccard_score(pd[:, i:i + 1, :, :], gt[:, i:i + 1, :, :]))
    score = sum(score) / n_classes
    return score


def get_accuracy_multi(pd, gt, n_classes):
    """
    params: pd==prediction
    params: gt: ground truth
    return: accuracy score
    """

    score = []
    for i in range(n_classes):
        score.append(get_accuracy(pd[:, i:i + 1, :, :], gt[:, i:i + 1, :, :]))
    score = sum(score) / n_classes
    return score


def get_sensitivity_multi(pd, gt, n_classes):
    """
    params: pd==prediction
    params: gt: ground truth
    return: sensitivity score / recall rate
    """

    score = []
    for i in range(n_classes):
        score.append(get_sensitivity(pd[:, i:i + 1, :, :], gt[:, i:i + 1, :, :]))
    score = sum(score) / n_classes
    return score


def get_specificity_multi(pd, gt, n_classes):
    score = []
    for i in range(n_classes):
        score.append(get_specificity(pd[:, i:i + 1, :, :], gt[:, i:i + 1, :, :]))
    score = sum(score) / n_classes
    return score

############################################################################## add by xfy

def threhold(inp, th=0.5):
    inp_ = np.copy(inp)
    inp_[inp_>th] = 1.
    inp_[inp_<=th] = 0.
    return inp_


class Metric(object):
    def __init__(self, opt):
        self.num_classes = opt.n_classes
        self.eye = np.eye(self.num_classes)

    @staticmethod
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        else:
            return data

    def onehot(self, labels):
        if labels.ndim == 2:
            return labels
        else:
            return self.eye[labels]

    @property
    def name(self):
        return self.__class__.__name__

    @staticmethod
    def de_onehot(label_matrix):
        if label_matrix.ndim == 1:
            return label_matrix
        else:
            return label_matrix.argmax(axis=1)

    def __call__(self):
        raise NotImplementedError


class Dice(Metric):
    def __init__(self, opt):
        super().__init__(opt)
        normalization = opt.get('normalization', 'sigmoid')
        assert normalization in ['sigmoid', 'softmax', 'none']

        self.num_classes = opt.n_classes
        self.ignore_index = 255

    def dice_2d(self, input, target):
        if input.ndim == target.ndim + 1:
            target = expand_as_one_hot(target, self.num_classes)

        return compute_per_channel_dice(input, target)

    def dice_3d(self, input, target):
        if input.ndim == target.ndim + 1:
            target = expand_as_one_hot(target, self.num_classes)

        return compute_per_channel_dice(input, target)

    def __call__(self, input, target):
        input = self.to_numpy(input)
        input = threhold(input)
        target = self.to_numpy(target)

        if input.ndim == 4:
            # N * X * Y
            return self.dice_2d(input, target)
        elif input.ndim == 5:
            # N * X * Y * Z
            return self.dice_3d(input, target)
        else:
            raise RuntimeError(
                f'The shape of target is {target.shape}.')


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    # TODO ignore index
    """
    Computes DiceCoefficient as defined in
        https://arxiv.org/abs/1606.04797 given  a multi channel
        input and target.
    Assumes the input is a normalized probability, e.g.
        a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.shape == target.shape, "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.astype(np.float)

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1)
    denominator = input.sum(-1) + target.sum(-1)
    return 2 * (intersect / denominator.clip(min=epsilon)).mean()


def flatten(array):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = array.shape[1]
    # new axis order
    axis_order = (1, 0) + tuple(range(2, array.ndim))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    # transposed = tensor.permute(axis_order)
    transposed = np.transpose(array, (axis_order))
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)


def expand_as_one_hot(input, C):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets
        converted to its corresponding one-hot vector. It is assumed that
        the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    # assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    # input = input.unsqueeze(1)
    input = np.expand_dims(input, axis=1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.shape)
    shape[1] = C

    # scatter to get the one-hot tensor
    # return torch.zeros(shape).to(input.device).scatter_(1, input, 1)
    result = np.zeros(shape)
    np.put_along_axis(result, input, 1, axis=1)
    return result


# add by xfy -----------------------------------------------

def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss =  (2 * intersection + eps) / (unionset + eps)

    return loss.sum() / N


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N