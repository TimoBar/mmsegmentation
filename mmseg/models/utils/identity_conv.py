from torch import nn
import torch

def set_identity_layer_mode(module, value):
    for n, p in module.named_modules():
        if isinstance(p, Identity):
            p.set_pruning_graph_mode(value)


class Identity(nn.Module):
    def __init__(self, out_channels, dim=0):
        self.dim = dim
        super().__init__()
        self.identity_conv = nn.Conv2d(out_channels, out_channels, (1, 1), bias=False, padding=(0, 0))
        self.init_itentity_convs()
        self.pruning_graph_mode = False
        self.non_zero_indexes = self.get_non_zero_feature_maps()
        self.out_channels = out_channels

    def get_non_zero_feature_maps(self):
        w = self.identity_conv.weight[:, :, 0, 0].sum(dim=1-self.dim)
        l = w.nonzero(as_tuple=True)[0].tolist()
        return l

    def set_pruning_graph_mode(self, value):
        self.pruning_graph_mode = value
        self.non_zero_indexes = self.get_non_zero_feature_maps()

    def init_itentity_convs(self):
        out_f, in_f, kx, ky = self.identity_conv.weight.size()
        self.identity_conv.weight = torch.nn.Parameter(
            torch.eye(out_f, in_f).cuda().expand(kx, ky, out_f, in_f).permute(2, 3, 0, 1), requires_grad=True)

    def forward(self, x):
        if self.pruning_graph_mode:
            return self.identity_conv(x)
        else:
            n, c, h, w = x.size()
            if self.dim == 0:
                res = torch.zeros((n, self.out_channels, h, w)).cuda()
                res[:, self.non_zero_indexes, :, :] = res[:, self.non_zero_indexes, :, :] + x
            else:
                res = x[:, self.non_zero_indexes, :, :]
            return res


class IdentityConv2d(Identity):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        conv_class=torch.nn.Conv2d,
        dim=0):
        super().__init__(out_channels, dim)
        self.conv = conv_class(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)

    def forward(self, x):
        if self.pruning_graph_mode:
            return self.identity_conv(self.conv(x))
        else:
            out = self.conv(x)
            n, c, h, w = out.size()
            res = torch.zeros((n, self.out_channels, h, w)).cuda()
            res[:,self.non_zero_indexes, :, :] = res[:,self.non_zero_indexes, :, :] + out
            return res
