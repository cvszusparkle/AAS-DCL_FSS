import torch
from thop import profile
from thop import clever_format

def caculate_net_size(net):
    x = torch.normal(0, 1, (1, 3, 448, 448))
    net.train()
    flops, params = profile(net, inputs=(x,))
    return clever_format([flops, params], "%.3f")

def num_params(net):
    return f"{sum(param.numel() for param in net.parameters())/10**6}M"

if __name__ == '__main__':
    # print(caculate_net_size(SegNet(3,1)))
    pass