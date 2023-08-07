from typing import Optional, Tuple
import functools

import torch
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

import torch
from torch import optim


def get_weighting_matrix(b, rows, columns):
    return torch.ones((rows, columns)) * b.unsqueeze(1).expand(rows, columns)

class StaticMaskLinear(nn.Module):
    def __init__(self, size_weight, size_bias, mask_dim=0, factor=1.0):
        super().__init__()
        self.factor = factor
        #self.permutation = None
        self.size_weight = size_weight
        self.size_bias = size_bias
        self.size_together = list(size_weight)

        self.size_together[-1] = self.size_weight[-1] + 1 # because of bias

        self.dim = mask_dim
        self.permutation_initialized = False

        self.pruning_size = self.size_weight[self.dim]
        self.p1 = torch.nn.parameter.Parameter(torch.ones((self.pruning_size), requires_grad=True)*self.factor)
        if self.dim == 1:
            self.non_pruning_size = self.size_together[0]
        else:
            self.non_pruning_size = self.size_together[1]

    def get_loss(self):
        return torch.abs(self.p1)
    def get_mask(self):
        return get_weighting_matrix(torch.min(self.p1 * self.factor, torch.tensor([self.pruning_size]).cuda()), self.pruning_size, self.non_pruning_size)[self.permutation]

class StaticMaskConv2d(StaticMaskLinear):
    def __init__(self, input_features, output_features, kernel_size, factor=100.0):
        size_weight = (output_features, kernel_size * kernel_size * input_features)
        size_bias = output_features
        super().__init__(size_weight, size_bias, mask_dim=0, factor=factor)
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size

    def get_mask(self):
        mask = super().get_mask()
        mask_weight = mask[:, 0:-1] = mask[:, 0:-1]
        mask_bias = mask[:, -1]
        return torch.reshape(mask_weight, (self.output_features, self.input_features, self.kernel_size, self.kernel_size)), mask_bias


def rename_parameter(obj, old_name, new_name):
    def rename_param(obj, old_name, new_name):
        #print(obj.__dict__.get('_parameters').keys())
        obj.__dict__.get('_parameters')[new_name] = obj._parameters.pop(old_name)
    pre, _, post = old_name.rpartition('.')
    pren, _, postn = new_name.rpartition('.')
    return rename_param(rgetattr(obj, pre) if pre else obj, post, postn)

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rdelattr(obj, attr, *args):
    def _delattr(obj, attr):
        return delattr(obj, attr, *args)
    return functools.reduce(_delattr, [obj] + attr.split('.'))

def mask_class_wrapper(super_class, mode="linear"):
    def init_wrapper(self, *args, **kw):
        super_class.__init__(self, *args, **kw)
        self.names = [name for name, _ in self.named_parameters()]
        self.masks = {}
        for i, name in enumerate(self.names):
            if mode=="linear":
                if "weight" in name.split(".")[-1] and i<len(self.names)-1 and "bias" in self.names[i+1].split(".")[-1]:
                    continue
                if "bias" in name.split(".")[-1] and i != 0 and "weight" in self.names[i - 1].split(".")[
                    -1]:
                    size_bias = rgetattr(self, name).size()
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    self.masks[name] = StaticMaskLinear(size_weight, size_bias)
                    self.masks[self.names[i - 1]] = self.masks[name]
            elif mode=="conv":
                if name.split(".")[-1] == "weight" and i<len(self.names)-1 and self.names[i+1].split(".")[-1] == "bias":
                    continue
                if name.split(".")[-1] == "weight" and not (i < len(self.names) - 1 and self.names[i + 1].split(".")[
                        -1] == "bias"):
                    size_weight = rgetattr(self, name).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    StaticMaskConv2d(feature_input, feature_output, kernel_size)
                    self.masks[name] = StaticMaskConv2d(feature_input, feature_output, kernel_size)
                if name.split(".")[-1] == "bias" and i != 0 and self.names[i - 1].split(".")[
                    -1] == "weight":
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    StaticMaskConv2d(feature_input, feature_output, kernel_size)
                    self.masks[name] = StaticMaskConv2d(feature_input, feature_output, kernel_size)
                    self.masks[self.names[i - 1]] = self.masks[name]
            else:
                raise NotImplementedError(f"Mode {mode} not implemented yet")
        self.module_list = nn.ModuleList(self.masks.values())

    def reset_parameters_wrapper(self, *args, **kw):
        return super_class._reset_parameters(self,*args, **kw)

    def get_mask_loss(self):
        loss = 0
        for mask in self.masks.values():
            loss = loss + mask.get_loss()
        return loss

    def train_wrapper(self, mode_train: bool = True, *args, **kw):
        return super_class.train(self, mode=mode_train, *args, **kw)

    def forward_wrapper(self, *args, **kw):
        if self.training:
            for i, name in enumerate(self.names):
                rename_parameter(self, name, name+"_")
                if mode == "linear":
                    mask = self.masks[name].get_mask()[:, 0:-1] if "weight" in name else self.masks[name].get_mask()[:, -1]
                elif mode == "conv":
                    mask = self.masks[name].get_mask()[0] if "weight" in name else self.masks[name].get_mask()[1]
                else:
                    raise NotImplementedError(f"Mode {mode} not implemented yet")
                rsetattr(self, name, rgetattr(self, name+"_") * mask)

            output = super_class.forward(self, *args, **kw)

            for name in self.names:
                rename_parameter(self, name+"_", name)
        else:
            output = super_class.forward(self, *args, **kw)

        return output

    return type(f"Wrapper{super_class}", (super_class, ), {
        # constructor
        "__init__": init_wrapper,

        # member functions
        "_reset_parameters": reset_parameters_wrapper,
        "get_mask_loss": get_mask_loss,
        "train": train_wrapper,
        "forward": forward_wrapper
    })

def get_p1_values(module):
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, StaticMaskLinear):
            p1_list[n] = p.p1
    return p1_list

def get_num_pruned(module):
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, StaticMaskLinear):
            p1_list[n] = (p.p1 * p.factor).int().float()
    return p1_list

def get_percentage_pruned(module):
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, StaticMaskLinear):
            p1_list[n] = (p.p1 * p.factor).int().float() / p.pruning_size
    return p1_list

def get_p1_loss(module):
    loss_sum = 0
    for n, p in module.named_modules():
        if isinstance(p, StaticMaskLinear):
            loss_sum = loss_sum + p.get_loss()
    return loss_sum
