import numpy as np
import torch.nn

from torch.nn.modules.module import register_module_forward_pre_hook
from mmseg.models.utils.mask_wrapper import rgetattr, rsetattr
import torch

def rhasattr(obj, attr):
    pre, _, post = attr.rpartition('.')
    return hasattr(rgetattr(obj, pre) if pre else obj, post)

# from https://stackoverflow.com/questions/69132963/delete-a-row-by-index-from-pytorch-tensor
# deletes the indexes in the delete_raw_list from the tensor x in the dimension delete_dim
def delete_tensor_row(x, delete_raw_list, delete_dim):
    index = np.array(range(x.size(delete_dim)))
    del_index = np.array(delete_raw_list)
    print(delete_raw_list)
    print(x.size())
    new_index = np.delete(index, del_index, axis=0)

    slicing_idx = []
    for i in range(x.dim()):
        slicing_idx.append(slice(None) if i != delete_dim else new_index)

    new_x = x[tuple(slicing_idx)]

    return new_x

def reduce_feature_dim(module, weight_name, bias_name, model, remove_indexes, example_input, max_tries=100):
    # register pre forward hooks:
    module_call_stack = []

    def hook(module, input):
        if isinstance(input, torch.Tensor):
            s = input.size()
        else:
            s = None
        module_call_stack.append((module, s, module.named_children()))

    handle = register_module_forward_pre_hook(hook)
    module_weight = rgetattr(module, weight_name)
    if rhasattr(module, bias_name):
        module_bias = rgetattr(module, bias_name)
    else:
        module_bias = None

    steps = []

    # reduce module weight and bias
    reduce_dim = module_weight.size()[0]
    steps.append((module, weight_name, module_weight.clone()))
    rsetattr(module, weight_name, torch.nn.Parameter(delete_tensor_row(module_weight, remove_indexes, 0)))
    reduce_dim_after = rgetattr(module, weight_name).size()[0]
    if hasattr(module, "out_channels"):
        module.out_channels = reduce_dim_after
    if module_bias is not None:
        steps.append((module, bias_name, module_bias.clone()))
        rsetattr(module, bias_name, torch.nn.Parameter(delete_tensor_row(module_bias, remove_indexes, 0)))

    # try to fix dimension of sucessor layers
    for i in range(max_tries):
        module_call_stack[:] = []
        try:
            model(example_input)
            handle.remove()
            print("Suceeded")
            return module_weight, module_bias
        except Exception as e:
            for i_size in module_call_stack:
                print(i_size[1], [n for n in i_size[2]])

            if module_call_stack[-1][0] != module:
                to_reduce = module_call_stack[-1][0]
            else:
                to_reduce = module_call_stack[-2][0]

            delete_dim = -1
            weight_name = ""
            for name, _ in to_reduce.named_parameters():
                if "weight" in name:
                    weight_name = name
                    break
            #print([name for name, _ in to_reduce.named_parameters()])
            print(rgetattr(to_reduce, weight_name).size())
            for i, s in enumerate(rgetattr(to_reduce, weight_name).size()):
                if s == reduce_dim:
                    print("Found reduce dim!!")
                    delete_dim = i
                    break
            if delete_dim == -1:
                print("Dimensions of next tensor doesn't match")
                #handle.remove()
                break

            steps.append((to_reduce, weight_name, rgetattr(to_reduce, weight_name).clone()))
            rsetattr(to_reduce, weight_name, torch.nn.Parameter(delete_tensor_row(rgetattr(to_reduce, weight_name), remove_indexes, delete_dim)))
            if hasattr(to_reduce, "in_channels"):
                to_reduce.in_channels = reduce_dim_after

    print("Failed to reduce weight and bias")
    print("Revert changes")
    for step in steps:
        module, param_name, oiginal_tensor = step
        rsetattr(module, param_name,
                 torch.nn.Parameter(oiginal_tensor))
    print("Changes reverted")
    handle.remove()
    return None