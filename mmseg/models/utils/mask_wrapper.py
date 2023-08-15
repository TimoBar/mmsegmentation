from typing import Optional, Tuple
import functools

import torch
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

import torch
from torch import optim



def logistic_function(x, k, b):
    # print(f"x: {x}, k: {k}, b: {b}, x-b: {x-b}, -k*(x-b): {-k * (x - b)}")
    #print(x.device, b.device)
    e = torch.clip(-k * (x - b), max=20.0)
    res = 1 / (1 + torch.exp(e))
    return res


def get_index_matrix(rows, colums):
    arr = torch.zeros((rows, colums), requires_grad=False).to(torch.float).cuda()
    for i in range(rows):
        arr[i] = i
    return arr


def get_weighting_matrix(b, rows, columns, k=3):
    return logistic_function(get_index_matrix(rows, 1), k, b).expand(rows, columns)


def calc_dummy_loss(weighting_matrix):
    return torch.sum(torch.abs(weighting_matrix), )


def get_permutation_vector(weight):
    return torch.argsort(torch.sum(torch.abs(weight), dim=1))

class LearnableMask(nn.Module):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

class LearnableMaskLinear(LearnableMask):
    def __init__(self, size_weight, size_bias, mask_dim=0, factor=100.0):
        super().__init__()
        self.p1 = torch.nn.parameter.Parameter(torch.tensor([-1 / factor], requires_grad=True).cuda())
        self.factor = factor
        # self.permutation = None
        self.size_weight = size_weight
        self.size_bias = size_bias
        self.size_together = list(size_weight)

        self.size_together[-1] = self.size_weight[-1] + 1  # because of bias

        self.dim = mask_dim
        self.permutation_initialized = False

        self.pruning_size = self.size_weight[self.dim]
        if self.dim == 1:
            self.non_pruning_size = self.size_together[0]
        else:
            self.non_pruning_size = self.size_together[1]

        self.register_buffer("permutation", torch.zeros(self.pruning_size).int().cuda())

    def reinit(self, size_weight, size_bias, mask_dim=0):
        dim_diff = self.pruning_size - size_weight[mask_dim]

        self.p1.copy_(self.p1 - dim_diff / self.factor)
        # self.permutation = None
        self.size_weight = size_weight
        self.size_bias = size_bias
        self.size_together = list(size_weight)
        self.size_together[-1] = self.size_weight[-1] + 1  # because of bias

        self.dim = mask_dim
        # self.permutation_initialized = False

        self.pruning_size = self.size_weight[self.dim]
        if self.dim == 1:
            self.non_pruning_size = self.size_together[0]
        else:
            self.non_pruning_size = self.size_together[1]
        # self.permutation = torch.zeros(self.pruning_size).int()

        if dim_diff > 0 and self.permutation_initialized:
            new_perm = self.permutation - dim_diff
            keep_rows = torch.where(new_perm < 0, 0, 1).nonzero(as_tuple=True)[0].tolist()
            new_perm = new_perm[keep_rows]
            zero_to_n = torch.argsort(torch.zeros_like(new_perm)).cuda()
            new_perm[torch.argsort(new_perm)] = zero_to_n
            self.permutation = new_perm
    def check_permutation_initialized(self):
        if not self.permutation_initialized and self.permutation.sum() == 0:
            return False
        elif not self.permutation_initialized:
            self.permutation = self.permutation.int()
            self.permutation_initialized = True
            return True
        elif self.permutation_initialized:
            return True

    def update_permutation(self, weight, bias):
        if not self.permutation_initialized:
            self.permutation_initialized = True

        # w = get_weighting_matrix(torch.min(self.p1, torch.tensor([self.pruning_size * 1.0])), self.pruning_size, self.non_pruning_size)
        wb = torch.hstack([weight, torch.unsqueeze(bias, -1)])
        self.permutation = get_permutation_vector(wb)

    def get_loss(self):
        # return 1/(torch.min(self.p1, torch.tensor([self.pruning_size])) + 5) - 0.001 * self.p1 * self.p1
        # return 1/(torch.min(self.p1, torch.tensor([self.pruning_size])) + 5) - 0.001 * self.p1 * self.p1
        return - torch.min(self.p1, torch.tensor([self.non_pruning_size]).cuda())

    def get_mask(self):
        mask = get_weighting_matrix(torch.min(self.p1 * self.factor, torch.tensor([self.pruning_size]).cuda()),
                                    self.pruning_size, self.non_pruning_size)[self.permutation]
        mask_weight = mask[:, 0:-1]
        mask_bias = mask[:, -1]
        return mask_weight, mask_bias


class LearnableMaskConv2d(LearnableMaskLinear):
    def __init__(self, input_features, output_features, kernel_size, factor=100.0):
        size_weight = (output_features, kernel_size * kernel_size * input_features)
        size_bias = output_features
        super().__init__(size_weight, size_bias, mask_dim=0, factor=factor)
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size

    def reinit(self, input_features, output_features, kernel_size):
        size_weight = (output_features, kernel_size * kernel_size * input_features)
        size_bias = output_features
        super().reinit(size_weight, size_bias)
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size

    def update_permutation(self, weight, bias):
        weight_shallow = torch.reshape(weight, (
            self.output_features, self.input_features * self.kernel_size * self.kernel_size))
        super().update_permutation(weight_shallow, bias)

    def get_mask(self):
        mask_weight, mask_bias = super().get_mask()
        return torch.reshape(mask_weight,
                             (self.output_features, self.input_features, self.kernel_size, self.kernel_size)), mask_bias


class LearnableMaskMHA(LearnableMaskLinear):
    def __init__(self, input_features, output_features, kernel_size, num_heads, factor=100.0):
        size_weight = (output_features // num_heads, kernel_size * kernel_size * input_features)
        size_bias = output_features // num_heads
        super().__init__(size_weight, size_bias // num_heads, mask_dim=0, factor=factor)
        self.input_features = input_features
        self.output_features = output_features // num_heads
        self.kernel_size = kernel_size
        self.num_heads = num_heads

    def reinit(self, input_features, output_features, kernel_size):
        size_weight = (output_features // self.num_heads, kernel_size * kernel_size * input_features)
        size_bias = output_features // self.num_heads
        super().reinit(size_weight, size_bias//self.num_heads)
        self.input_features = input_features
        self.output_features = output_features // self.num_heads
        self.kernel_size = kernel_size

    def update_permutation(self, weight, bias):
        weight_shallow = torch.abs(weight.reshape(self.num_heads, self.output_features, self.input_features * self.kernel_size * self.kernel_size)).sum(dim=0)
        bias_shallow = torch.abs(bias.reshape(self.num_heads, self.output_features)).sum(dim=0)
        super().update_permutation(weight_shallow, bias_shallow)

    def get_mask(self):
        mask_weight, mask_bias = super().get_mask()
        r = torch.reshape(mask_weight,
                             (self.output_features, self.input_features, self.kernel_size, self.kernel_size))
        weight_mask_r = r.expand(self.num_heads, self.output_features, self.input_features, self.kernel_size, self.kernel_size).reshape(self.num_heads * self.output_features, self.input_features, self.kernel_size, self.kernel_size)
        bias_mask_r = mask_bias.expand(self.num_heads, self.output_features).reshape(self.num_heads * self.output_features)
        return weight_mask_r, bias_mask_r

class LearnableMaskMHALinear(LearnableMaskLinear):
    def __init__(self, size_weight, size_bias, num_heads, mask_dim=0, factor=100.0):
        output_features, input_features = size_weight
        size_weight = (output_features // num_heads, input_features)
        self.num_heads = num_heads
        super().__init__(size_weight, size_bias // num_heads, mask_dim, factor)

    def reinit(self, size_weight, size_bias, mask_dim=0):
        output_features, input_features = size_weight
        size_weight = (output_features // self.num_heads, input_features)
        size_bias = int(size_bias) // self.num_heads
        super().reinit(size_weight, size_bias)

    def update_permutation(self, weight_in, bias_in, weight_out, bias_out):
        weight_shallow_in = torch.abs(torch.abs(
            weight_in.reshape(3, self.pruning_size * self.num_heads, self.non_pruning_size - 1)).sum(dim=0).reshape(
            self.num_heads, self.pruning_size, self.non_pruning_size - 1)).sum(dim=0)
        weight_shallow_out = torch.abs(
            weight_out.reshape(self.num_heads, self.pruning_size, self.non_pruning_size - 1)).sum(dim=0)
        bias_shallow_in = torch.abs(torch.abs(bias_in.reshape(3, self.num_heads*self.pruning_size)).sum(dim=0).reshape(self.num_heads, self.pruning_size)).sum(dim=0)
        bias_shallow_out = torch.abs(bias_out.reshape(self.num_heads, self.pruning_size)).sum(dim=0)
        super().update_permutation(weight_shallow_in+weight_shallow_out, bias_shallow_in+bias_shallow_out)

    def get_mask(self):
        mask_weight, mask_bias = super().get_mask()
        out_proj_weight_mask = mask_weight.expand(self.num_heads, self.pruning_size, self.non_pruning_size-1).reshape(self.num_heads * self.pruning_size, self.non_pruning_size-1)
        out_proj_bias_mask = mask_bias.expand(self.num_heads, self.pruning_size).reshape(self.num_heads * self.pruning_size)
        in_proj_weight_mask = out_proj_weight_mask.expand(3, self.num_heads * self.pruning_size, self.non_pruning_size-1).reshape(3 * self.num_heads * self.pruning_size, self.non_pruning_size-1)
        in_proj_bias_mask = out_proj_bias_mask.expand(3, self.num_heads * self.pruning_size).reshape(3 * self.num_heads * self.pruning_size)
        return in_proj_weight_mask, in_proj_bias_mask, out_proj_weight_mask, out_proj_bias_mask

class LearnableKernelMask(LearnableMask):
    def get_weighting_matrix(self, b, feature_maps, pruning_dim, non_pruning_dim, k=7):
        b_expand = b.unsqueeze(1).unsqueeze(1).expand((feature_maps, pruning_dim, non_pruning_dim))
        index_mat = self.get_index_matrix(pruning_dim, non_pruning_dim)
        return logistic_function(index_mat.unsqueeze(0).expand(feature_maps, pruning_dim, non_pruning_dim), k, b_expand)

    def get_index_matrix(self, rows, colums):
        if rows % 2 == 1:
            arr = torch.zeros((rows, colums), requires_grad=False).to(torch.float).cuda()
            for i in range(rows):
                arr[i] = (rows // 2 - abs(rows // 2 - i)) * 2
            for i in range(rows - rows // 2 - 1):
                arr[i] = arr[i] + 1
            return arr
        else:
            arr = self.get_index_matrix(rows + 1, colums)[:-1, :]
            for i in range(rows - rows // 2):
                arr[i + rows // 2] = arr[i + rows // 2] - 2
            return arr

    def __init__(self, input_features, output_features, kernel_size, factor=100.0, loss_factor=10.0):
        super().__init__()
        size_weight = (output_features, kernel_size * kernel_size * input_features)
        size_bias = output_features
        self.p1 = torch.nn.parameter.Parameter(-1*torch.ones((output_features * 2)).cuda() / factor, requires_grad=True)
        self.factor = factor
        self.size_weight = size_weight
        self.size_bias = size_bias
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size
        self.loss_factor = loss_factor
        self.permutation = True
        self.permutation_initialized = True

    def check_permutation_initialized(self):
        return True

    def update_permutation(self, weight, bias):
        pass

    def get_loss(self):
        return - torch.sum(self.p1) #/ (self.kernel_size**2) * self.loss_factor

    def get_mask(self):
        b_x = self.p1[0:self.p1.size(0) // 2] * self.factor
        b_y = self.p1[self.p1.size(0) // 2:] * self.factor
        m_x = self.get_weighting_matrix(b_x, self.output_features, self.kernel_size,
                                        self.kernel_size * self.input_features, k=3).reshape(
            (self.output_features, self.kernel_size, self.input_features, self.kernel_size)).permute((0, 2, 1, 3))
        m_y = self.get_weighting_matrix(b_y, self.output_features, self.kernel_size,
                                        self.kernel_size * self.input_features, k=3).reshape(
            (self.output_features, self.kernel_size, self.input_features, self.kernel_size)).permute((0, 2, 3, 1))

        mask = m_x * m_y
        mask_bias = mask[:, 0, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2]
        return mask, mask_bias


def rename_parameter(obj, old_name, new_name):
    def rename_param(obj, old_name, new_name):
        # print(obj.__dict__.get('_parameters').keys())
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


def mask_class_wrapper(super_class, mode="linear", embedded_dims=64):
    def init_masks(self):
        self.masks = {}
        for i, name in enumerate(self.names):
            if "bn." in name:
                continue
            # If current name is weight and the next name in the list is not called bias
            # -> weight with no bias
            if "weight" in name.split(".")[-1] and not (
                    i < len(self.names) - 1 and self.names[i + 1].split(".")[-1] == "bias"):
                if mode == "conv":
                    size_weight = rgetattr(self, name).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableMaskConv2d(feature_input, feature_output, kernel_size)
                elif mode == "kernel":
                    size_weight = rgetattr(self, name).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableKernelMask(feature_input, feature_output, kernel_size)
                elif mode == "mha_conv":
                    size_weight = rgetattr(self, name).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableMaskMHA(feature_input, feature_output, kernel_size, feature_output//embedded_dims)
                elif mode == "mha_linear":#out_proj.weight, out_proj.bias, in_proj_weight, in_proj_bias
                    size_weight = rgetattr(self, name).size()
                    size_bias = size_weight[0]
                    if "out_proj.weight" in name:
                        self.masks[name] = LearnableMaskMHALinear(size_weight, int(size_bias), int(size_bias//embedded_dims))
                        self.masks["in_proj_weight"] = self.masks[name]
            # If current name is bias and the previous name in the list was weight
            # -> weight and bias belong together
            elif "bias" in name.split(".")[-1] and i != 0 and "weight" in self.names[i - 1].split(".")[
                -1]:
                if mode == "linear":
                    size_bias = rgetattr(self, name).size()
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    self.masks[name] = LearnableMaskLinear(size_weight, size_bias)
                    self.masks[self.names[i - 1]] = self.masks[name]
                elif mode == "conv":
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableMaskConv2d(feature_input, feature_output, kernel_size)
                    self.masks[self.names[i - 1]] = self.masks[name]
                elif mode == "kernel":
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableKernelMask(feature_input, feature_output, kernel_size)
                    self.masks[self.names[i - 1]] = self.masks[name]
                elif mode == "mha_conv":
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableMaskMHA(int(feature_input), int(feature_output), int(kernel_size), int(feature_output)//embedded_dims)
                    self.masks[self.names[i - 1]] = self.masks[name]
                elif mode == "mha_linear":
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    size_bias = size_weight[0]
                    if "out_proj.weight" in self.names[i - 1]:
                        self.masks[name] = LearnableMaskMHALinear(size_weight, int(size_bias), int(size_bias//embedded_dims))
                        self.masks["in_proj_weight"] = self.masks[name]
                        self.masks[self.names[i - 1]] = self.masks[name]
                        self.masks["in_proj_bias"] = self.masks[name]
                else:
                    raise NotImplementedError(f"Mode {mode} not implemented yet")

        self.module_list = nn.ModuleList(self.masks.values())
    def delete_masks(self):
        for mask in self.masks.values():
            del mask
        del self.module_list
        self.masks = {}
    def reinit_masks(self):
        with torch.no_grad():
            for i, name in enumerate(self.names):
                if name in self.masks and "weight" in name:
                    mask = self.masks[name]
                    if mode == "conv" or mode == "mha_conv":
                        size_weight = rgetattr(self, name).size()
                        feature_output, feature_input, kernel_size, _ = size_weight
                        mask.reinit(feature_input, feature_output, kernel_size)
                    elif mode == "linear" or (mode == "mha_linear" and "out_proj.weight" in name):
                        size_weight = rgetattr(self, name).size()
                        size_bias = size_weight[0]
                        mask.reinit(size_weight, size_bias)


    def init_wrapper(self, *args, **kw):
        super_class.__init__(self, *args, **kw)
        self.mask_class_wrapper = True
        self.names = [name for name, _ in self.named_parameters()]
        self.init_masks()

    def reset_parameters_wrapper(self, *args, **kw):
        return super_class._reset_parameters(self, *args, **kw)

    def get_mask_loss(self):
        loss = 0
        for mask in self.masks.values():
            loss = loss + mask.get_loss()
        return loss

    def train_wrapper(self, mode_train: bool = True, *args, **kw):
        super_class.train(self, mode=mode_train, *args, **kw)
        if not mode_train:
            with torch.no_grad():
                for i, name in enumerate(self.names):
                    if name in self.masks:
                        weight_idx, bias_idx = 0, 1
                        if mode == "mha_linear":
                            weight_idx, bias_idx = (0, 1) if "in_proj" in name else (2, 3)

                        mask = self.masks[name].get_mask()[weight_idx] if "weight" in name else self.masks[name].get_mask()[bias_idx]
                        mask = torch.where(mask < 0.001, 0.0, 1.0)
                        param = rgetattr(self, name)
                        if param.size() != mask.size():
                            print("err")
                            mask = self.masks.pop(name)
                            self.names.pop(i)
                            del mask
                        else:
                            param.copy_(param * mask)
                        # remove_mask = torch.where(mask[:,] == 0.0, 1.0, 0.0)

        return self

    def initialize_permutation_if_empty(self, name, index):
        if not self.masks[name].check_permutation_initialized():
            # If current name is weight and next is bias
            # -> weight and bias belong together
            if "in_proj_weight" in name:
                self.masks[name].update_permutation(rgetattr(self, "in_proj_weight_"), rgetattr(self, "in_proj_bias"),
                                                    rgetattr(self, "out_proj.weight"), rgetattr(self, "out_proj.bias"))
            elif "weight" in name.split(".")[-1] and index < len(self.names) - 1 and "bias" in \
                    self.names[index + 1].split(".")[
                        -1]:
                self.masks[name].update_permutation(rgetattr(self, name + "_"),
                                                    rgetattr(self, self.names[index + 1]))
            # If current name is weight and next is not bias
            # -> weight without bias
            elif "weight" in name.split(".")[-1] and not (
                    index < len(self.names) - 1 and "bias" in self.names[index + 1].split(".")[
                -1]):
                self.masks[name].update_permutation(rgetattr(self, name + "_"),
                                                    torch.zeros((rgetattr(self, name + "_").size()[0])).cuda())

    def forward_wrapper(self, *args, **kw):
        if self.training:
            # print("forward called")
            for i, name in enumerate(self.names):
                if name in self.masks:
                    # rename parameter
                    rename_parameter(self, name, name + "_")

                    # init permutation if not initialized already
                    self.initialize_permutation_if_empty(name, i)

                    # get mask
                    weight_idx, bias_idx = 0, 1
                    if mode == "mha_linear":
                        weight_idx, bias_idx = (0, 1) if "in_proj" in name else (2, 3)
                    mask = self.masks[name].get_mask()[weight_idx] if "weight" in name else self.masks[name].get_mask()[bias_idx]

                    # create mask parameter with original name
                    rsetattr(self, name, rgetattr(self, name + "_") * mask)

            output = super_class.forward(self, *args, **kw)


            # rename parameters back
            for name in self.names:
                if name in self.masks:
                    rename_parameter(self, name + "_", name)
        else:

            for i, name in enumerate(self.names):
                if name in self.masks:
                    rename_parameter(self, name, name + "_")
                    self.initialize_permutation_if_empty(name, i)
                    rename_parameter(self, name + "_", name)
            output = super_class.forward(self, *args, **kw)

        return output

    return type(f"Wrapper{super_class}", (super_class,), {
        # constructor
        "__init__": init_wrapper,

        # member functions
        "_reset_parameters": reset_parameters_wrapper,
        "get_mask_loss": get_mask_loss,
        "train": train_wrapper,
        "forward": forward_wrapper,
        "initialize_permutation_if_empty": initialize_permutation_if_empty,
        "init_masks": init_masks,
        "delete_masks": delete_masks,
        "reinit_masks": reinit_masks
    })


def get_p1_values(module):
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            p1_list[n] = p.p1
    return p1_list


def get_num_pruned(module):
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, LearnableKernelMask):
            p1 = torch.min(torch.max(torch.zeros_like(p.p1).int(), (p.p1 * p.factor).int()), torch.ones_like(p.p1) * p.kernel_size)
            p1_x = p1[0:p1.size(0) // 2]
            p1_y = p1[p1.size(0) // 2:]
            his = torch.histogramdd(torch.stack([p1_x, p1_y]).permute(1, 0).cpu(),
                                    bins=[p.kernel_size + 1, p.kernel_size + 1],
                                    range=[0, p.kernel_size, 0, p.kernel_size])
            for row_idx in range(p.kernel_size + 1):
                if row_idx == p.kernel_size:
                    if not n + f"--0x0" in p1_list:
                        p1_list[n + f"--0x0"] = 0
                    p1_list[n + f"--0x0"] = p1_list[n + f"--0x0"] + torch.sum(his.hist[row_idx, 0:-1])
                else:
                    num = p.kernel_size - row_idx
                    p1_list[n + f"--{num}xN"] = his.hist[row_idx, 0:-1]

            p1_list[n + f"--0x0"] = p1_list[n + f"--0x0"] + torch.sum(his.hist[:, -1])
        elif isinstance(p, LearnableMask):
            p1_list[n] = (p.p1 * p.factor).int().float()
    return p1_list


def get_percentage_pruned(module):
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            p1_list[n] = (p.p1 * p.factor).int().float() / p.pruning_size
    return p1_list


def get_p1_loss(module):
    loss_sum = 0
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            loss_sum = loss_sum + p.get_loss()
    return loss_sum
