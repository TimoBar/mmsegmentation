import numpy as np
import torch.nn

from torch.nn.modules.module import register_module_forward_pre_hook
import torch

# from https://stackoverflow.com/questions/69132963/delete-a-row-by-index-from-pytorch-tensor
# deletes the indexes in the delete_raw_list from the tensor x in the dimension delete_dim
def delete_tensor_row(x, delete_raw_list, delete_dim):
    index = np.array(range(x.size(delete_dim)))
    del_index = np.array(delete_raw_list)
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
        module_call_stack.append(module)

    handle = register_module_forward_pre_hook(hook)
    module_weight = getattr(module, weight_name)
    module_bias = getattr(module, bias_name, None)

    # reduce module weight and bias
    reduce_dim = module_weight.size()[0]
    setattr(module, weight_name, torch.nn.Parameter(delete_tensor_row(module_weight, remove_indexes, 0)))
    reduce_dim_after = getattr(module, weight_name).size()[0]
    if hasattr(module, "out_channels"):
        module.out_channels = reduce_dim_after
    if module_bias is not None:
        setattr(module, bias_name, torch.nn.Parameter(delete_tensor_row(module_bias, remove_indexes, 0)))

    # try to fix dimension of sucessor layers
    for i in range(max_tries):
        module_call_stack[:] = []
        try:
            model(example_input)
            handle.remove()
            return module_weight, module_bias
        except Exception as e:
            if module_call_stack[-1] != module:
                to_reduce = module_call_stack[-1]
            else:
                to_reduce = module_call_stack[-2]

            delete_dim = -1
            for i, s in enumerate(to_reduce.weight.size()):
                if s == reduce_dim:
                    delete_dim = i
                    break
            if i == -1:
                print("Dimensions of next tensor doesn't match")
                handle.remove()
                return None

            to_reduce.weight = torch.nn.Parameter(delete_tensor_row(to_reduce.weight, remove_indexes, delete_dim))
            if hasattr(to_reduce, "in_channels"):
                to_reduce.in_channels = reduce_dim_after

    print("Failed to reduce weight and bias")
    handle.remove()
    return None