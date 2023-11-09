import functools

import torch
from torch import nn


def logistic_function(x, k, b, epsilon=5e-5):
    """

    Args:
        x: the input tensor
        k: the exponent
        b: the shift on the x-axis
        epsilon:

    Returns:

    """
    # use clip to make it numerically stable
    e = torch.clip(-k * (x - b), max=10.0, min=-10.0)
    res = 1 / (1 + torch.exp(e))
    res[res != torch.clamp(res, epsilon)] = 0
    return res


def get_index_matrix(rows, colums, device="cuda"):
    """
    creates a 2D Tensor of the following form
    [[0, 0, 0, 0, ...]
    [1, 1, 1, 1, ...]
    [2, 2, 2, 2, ...]
    ...
    [n, n, n, n, ...]]

    Args:
        rows: number of rows
        colums: number of columns
        device: the device to save the created tensor to

    Returns:

    """
    arr = torch.zeros((rows, colums), requires_grad=False, device=device, dtype=torch.float)
    for i in range(rows):
        arr[i] = i
    return arr


def get_weighting_matrix(b, rows, columns, k=7, device="cuda"):
    """
    creates a matrix of the following form:
        [[0, 0, 0, 0, ...]
        [1, 1, 1, 1, ...]
        [2, 2, 2, 2, ...]
        ...
        [n, n, n, n, ...]]
    and applies to every element x in the matrix a sigmoid/logistic function of the form:
        f(x) = 1 / (1 + e^(-k * (x - b)))

    Args:
        b: the shift on the x-axis
        rows: number of rows
        columns: number of columns
        k: parameter that defines slope of the sigmoid/logistic function
        device:

    Returns:

    """
    res = logistic_function(get_index_matrix(rows, 1, device), k, b).expand(rows, columns)
    return res



def get_permutation_vector(weight):
    """
    gets the permutation vector by sorting the number of neurons in ascending order.
    The weight matrix is summed over in_size (dim=1) and ordered over out_size (dim=0)

    The function is the simplest estimation of an importance ranking of the neurons in a linear layer

    Args:
        weight: the weight matrix of size (out_size x in_size)

    Returns:
        the permutation vector in ascending order

    """
    return torch.argsort(torch.sum(torch.abs(weight), dim=1))


class LearnableMask(nn.Module):
    """
    Base class for the learnable masks for structured pruning
    """
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)


class LearnableMaskLinear(LearnableMask):
    """
    Learnable Mask for a linear layer
    """

    def __init__(self, size_weight, size_bias, mask_dim=0, lr_mult_factor=100.0, k=7):
        """

        Args:
            size_weight: size of the weight to mask out as a 2d tensor/tuple of the form (out_features, in_features)
            size_bias: size of the bias as a 1d tensor
            mask_dim: the dimension to mask out, default=0 -> out_features
            lr_mult_factor: the parameter p1 is divided by lr_mult_factor and for generating the mask it is multiplied back by lr_mult_factor,
                            theoretically this parameter should not change anything, but it does.
                            The reason for this is that the effect of weight decay is responsible for this
                            TODO: remove lr_mult_factor by turning off weight decay for p1
            k: the exponent for the sigmoid function for mask creation f(x) = 1 / (1 + e^(-k * (x - b))), default = 7
        """
        super().__init__()
        self.k = k
        # a single parameter which is the only learnable scalar in this class, which can be
        # in the range [-1, pruning_size]
        self.p1 = torch.nn.parameter.Parameter(torch.tensor([-1 / lr_mult_factor], requires_grad=True))
        self.lr_mult_factor = lr_mult_factor
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

        # register permutation as this is not learnable, but must be saved with the model
        self.register_buffer("permutation", torch.zeros(self.pruning_size, device=self.get_device()).int())

    def get_device(self):
        """
        Getter for device of mask
        Returns: device

        """
        return self.p1.device

    def reinit(self, size_weight, size_bias, mask_dim=0):
        """
        Reinit masks after the linear layer was pruned, aka. some neurons have been removed.
        The mask parameter must be reduced by the number of neurons that have been removed from this layer

        Args:
            size_weight: new size of weight in linear layer after pruning - (new_out_features, in_features)
            size_bias: tensor of size new_out_features
            mask_dim: the dimension to mask out, default=0 -> out_features

        Returns:
            None
        """

        # get the number of removed neurons
        dim_diff = self.pruning_size - size_weight[mask_dim]

        # reduce the mask parameter by the amount dim_diff/self.lr_mult_factor
        self.p1.copy_(self.p1 - dim_diff / self.lr_mult_factor)

        # set class parameters to new sizes
        self.size_weight = size_weight
        self.size_bias = size_bias
        self.size_together = list(size_weight)
        self.size_together[-1] = self.size_weight[-1] + 1  # +1 is because of the bias

        self.dim = mask_dim

        self.pruning_size = self.size_weight[self.dim]
        if self.dim == 1:
            self.non_pruning_size = self.size_together[0]
        else:
            self.non_pruning_size = self.size_together[1]

        # fix permutation:
        # if the previous permutation was for example [2,5,1,3,0,4] and the first 3 channels have been removed
        # the resulting permutation should not anymore contain the previous entries for 0,1,2 -> [5,3,4]
        # Then the permutation must be corrected to start again with 0
        if dim_diff > 0 and self.permutation_initialized:
            new_perm = self.permutation - dim_diff
            keep_rows = torch.where(new_perm < 0, 0, 1).nonzero(as_tuple=True)[0].tolist()
            new_perm = new_perm[keep_rows]
            zero_to_n = torch.argsort(torch.zeros_like(new_perm)).to(self.get_device()).to(new_perm.dtype)
            new_perm[torch.argsort(new_perm)] = zero_to_n
            self.permutation = new_perm


    def check_permutation_initialized(self):
        """
        Check whether the permutation is already initialized.
        The permutation should not be initialized twice, because this would change the output

        Returns:
            True if initialized
        """
        if not self.permutation_initialized and self.permutation.sum() == 0:
            return False
        elif not self.permutation_initialized:
            self.permutation = self.permutation.int()
            self.permutation_initialized = True
            return True
        elif self.permutation_initialized:
            return True

    def update_permutation(self, weight, bias):
        """
        updates the permutation

        Args:
            weight: weight of the linear layer with shape (out_features, in_features)
            bias: bias of the linear layer with shape (out_features)

        Returns:
            return permutation vector with shape (out_features)
        """
        if not self.permutation_initialized:
            self.permutation_initialized = True

        # create one single vector of shape (out_features, in_features + 1)
        wb = torch.hstack([weight, torch.unsqueeze(bias, -1)])

        # get permutation
        self.permutation = get_permutation_vector(wb)

    def get_loss(self):
        """
        Simple loss function: loss = -p1

        To avoid p1 from getting unnecessary large we introduced a min term to prevent p1 from getting 5 larger
        than the number of neurons in this layer:
            loss = -min(p1, (self.pruning_size+5) / self.lr_mult_factor)
        Returns:
            loss
        """
        return - torch.min(self.p1, torch.tensor([(self.pruning_size + 5) / self.lr_mult_factor], device=self.get_device()))

    def get_mask(self):
        """
        get the mask based in the current parameter p1
        Returns:
            weighting mask
        """
        mask = get_weighting_matrix(torch.min(self.p1 * self.lr_mult_factor, torch.tensor([self.pruning_size + 5], device=self.get_device())),
                                    self.pruning_size, self.non_pruning_size, k=self.k, device=self.get_device())[self.permutation]
        mask_weight = mask[:, 0:-1]
        mask_bias = mask[:, -1]
        return mask_weight, mask_bias


class LearnableMaskConv2d(LearnableMaskLinear):
    """
    Learnable Mask for a Conv2d layer

    Basically very similar to the LearnableMaskLinear class, because the Conv2d
    layer is reshaped to work with the LearnableMaskLinear.

    The corresponding Conv2d(in_channels, out_channels, (kernel_size_x, kernel_size_y)) layer
    has a weight of shape (out_channels, in_channels, kernel_size_x, kernel_size_y). We want
    to mask out complete output channels, so we can reshape this weight
    to (out_channels, in_channels*kernel_size_x*kernel_size_y). Then we can easily apply again the same
    mask as for the LearnableMaskLinear for size size_weight=(out_channels, in_channels*kernel_size_x*kernel_size_y)
    """
    def __init__(self, input_features, output_features, kernel_size, lr_mult_factor=100.0, k=7):
        """

        Args:
            input_features: number input_features of Conv layer
            output_features: number output_features of Conv layer
            kernel_size: the kernel size of Conv layer
            lr_mult_factor: lr_mult_factor: the parameter p1 is divided by lr_mult_factor and for generating the mask it is multiplied back by lr_mult_factor,
                            theoretically this parameter should not change anything, but it does.
                            The reason for this is that the effect of weight decay is responsible for this
                            TODO: remove lr_mult_factor by turning off weight decay for p1
            k: the exponent for the sigmoid function for mask creation f(x) = 1 / (1 + e^(-k * (x - b))), default = 7
        """
        size_weight = (output_features, kernel_size * kernel_size * input_features)
        size_bias = output_features
        super().__init__(size_weight, size_bias, mask_dim=0, lr_mult_factor=lr_mult_factor, k=k)
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size

    def reinit(self, input_features, output_features, kernel_size):
        """
        Reinit masks after the linear layer was pruned, aka. some feature maps have been removed.
        The mask parameter must be reduced by the number of feature maps that have been removed from this layer
        Args:
            input_features: number of input_features
            output_features: new number of output_features
            kernel_size: kernel size

        Returns:
            None
        """

        size_weight = (output_features, kernel_size * kernel_size * input_features)
        size_bias = output_features
        super().reinit(size_weight, size_bias)
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size

    def update_permutation(self, weight, bias):
        """
        updates the permutation by calling the update_permutation function from the base class
        Args:
            weight:
            bias:

        Returns:

        """
        weight_shallow = torch.reshape(weight, (
            self.output_features, self.input_features * self.kernel_size * self.kernel_size))
        super().update_permutation(weight_shallow, bias)

    def get_mask(self):
        """
        get the mask based in the current parameter p1 by calling the get_mask function from the base class
        and reshaping it into the needed shape for the Conv2d layer
        Returns:
            weighting mask
        """
        mask_weight, mask_bias = super().get_mask()
        return torch.reshape(mask_weight,
                             (self.output_features, self.input_features, self.kernel_size, self.kernel_size)), mask_bias



class LearnableMaskMHALinear(LearnableMaskLinear):
    """
    Learnable Mask for a Multihead attention layer

    Basically very similar to the LearnableMaskLinear class, because the MultiHeadAttention
    layer is reshaped to work with the LearnableMaskLinear.

    The corresponding MultiHeadAttention(embed_dim, num_heads) layer
    has weights of shape:
        in_proj_weight:     (3*embed_dim, embed_dim)
        in_proj_bias:       (3*embed_dim)
        out_proj_weight:    (embed_dim, embed_dim)
        out_proj_bias:      (embed_dim)

    It is only possible to mask out a complete embed_dim dimension in every head of the num_heads.
    So there are embed_dim//num_heads elements that can be masked out.

    """

    def __init__(self, size_weight, size_bias, num_heads, mask_dim=0, lr_mult_factor=100.0, k=7):
        """

        Args:
            size_weight: the size of the weight of the corresponding MultiHeadAttention layer in (embed_dim, embed_dim)
            size_bias: the size of the bias of the corresponding MultiHeadAttention layer (embed_dim)
            num_heads: the number of heads in the corresponding MultiHeadAttention layer
            mask_dim: must be 0 and is just to match signature of base class
            lr_mult_factor: lr_mult_factor: the parameter p1 is divided by lr_mult_factor and for generating the mask it is multiplied back by lr_mult_factor,
                            theoretically this parameter should not change anything, but it does.
                            The reason for this is that the effect of weight decay is responsible for this
                            TODO: remove lr_mult_factor by turning off weight decay for p1
            k: the exponent for the sigmoid function for mask creation f(x) = 1 / (1 + e^(-k * (x - b))), default = 7
        """
        output_features, input_features = size_weight
        # We handle the mask creation of the MHA layer as a LinearLearnableMask of size (embed_dim//num_heads, embed_dim)
        size_weight = (output_features // num_heads, input_features)
        self.num_heads = num_heads
        super().__init__(size_weight, size_bias // num_heads, mask_dim, lr_mult_factor, k=k)

    def reinit(self, size_weight, size_bias, mask_dim=0):
        """
        Reinit masks after the MHA layer was pruned, aka. some embedded dimensions have been removed.
        The mask parameter must be reduced by the number of embedded dimensions that have been removed from this layer
        Args:
            size_weight: (embed, embed)
            size_bias: (embed)
            mask_dim: must be 0 and is just to match signature of base class

        Returns:
            None
        """
        output_features, input_features = size_weight
        size_weight = (output_features // self.num_heads, input_features)
        size_bias = int(size_bias) // self.num_heads
        super().reinit(size_weight, size_bias)


    def update_permutation(self, weight_in, bias_in, weight_out, bias_out):
        """
        updates the permutation by calling the update_permutation function from the base class
        Args:
            weight_in: the in_proj_weight of the MHA layer. Shape (3*embed_dim, embed_dim)
            bias_in: the in_proj_bias of the MHA layer. Shape (3*embed_dim)
            weight_out: the out_proj_weight of the MHA layer. Shape (embed_dim, embed_dim)
            bias_out: the out_proj_bias of the MHA layer. Shape (embed_dim)

        Returns:

        """
        embed_dim = self.pruning_size * self.num_heads
        # weight_in (3*embed_dim, embed_dim) is first reshaped to (3, embed_dim, embed_dim) and then summed up over
        # the first dim -> (embed_dim, embed_dim)
        # then the result is reshaped to (num_heads, embed_dim//num_heads, embed_dim) and then summed up again over
        # the first dim -> (embed_dim//num_heads, embed_dim)
        weight_shallow_in = torch.abs(torch.abs(weight_in.reshape(3, embed_dim, embed_dim)).sum(dim=0).reshape(
            self.num_heads, self.pruning_size, embed_dim)).sum(dim=0)

        # weight_out (embed_dim, embed_dim)) is reshaped to (num_heads, embed_dim//num_heads, embed_dim) and then summed up
        # over the first dim -> (embed_dim//num_heads, embed_dim)
        weight_shallow_out = torch.abs(
            weight_out.reshape(self.num_heads, self.pruning_size, embed_dim)).sum(dim=0)

        # bias_in (3*embed_dim) is first reshaped to (3, embed_dim) and then summed up over
        # the first dim -> (embed_dim)
        # then the result is reshaped to (num_heads, embed_dim//num_heads) and then summed up again over
        # the first dim -> (embed_dim//num_heads)
        bias_shallow_in = torch.abs(torch.abs(bias_in.reshape(3, self.num_heads*self.pruning_size)).sum(dim=0).reshape(self.num_heads, self.pruning_size)).sum(dim=0)

        # bias_out (embed_dim)) is reshaped to (num_heads, embed_dim//num_heads) and then summed up
        # over the first dim -> (embed_dim//num_heads)
        bias_shallow_out = torch.abs(bias_out.reshape(self.num_heads, self.pruning_size)).sum(dim=0)

        # call update_permutation from parent class with calculated sums
        super().update_permutation(weight_shallow_in+weight_shallow_out, bias_shallow_in+bias_shallow_out)

    def get_mask(self):
        """
        get the mask based in the current parameter p1 by calling the get_mask function from the base class
        and reshaping it into the needed shape for the MHA layer
        Returns:
            weighting mask
        """
        mask_weight, mask_bias = super().get_mask()
        # mask_weight has shape (embed_dim//num_heads, embed_dim)
        # mask_bias has shape (embed_dim//num_heads)

        embed_dim = self.non_pruning_size-1

        # expand mask_weight from (embed_dim//num_heads, embed_dim) to (embed_dim, embed_dim) by repeating it num_heads times
        out_proj_weight_mask = mask_weight.expand(self.num_heads, self.pruning_size, embed_dim).reshape(embed_dim, embed_dim)
        # for the out_proj_weight both dimensions need to be multiplied by the mask, so we need the transpose
        transpose = out_proj_weight_mask.permute(1, 0)
        out_proj_weight_mask = out_proj_weight_mask * transpose

        # expand mask_bias from (embed_dim//num_heads) to (embed_dim) by repeating it num_heads times
        out_proj_bias_mask = mask_bias.expand(self.num_heads, self.pruning_size).reshape(embed_dim)


        # expand out_proj_weight_mask from (embed_dim, embed_dim) to (3, embed_dim, embed_dim) by repeating it 3 times
        in_proj_weight_mask = out_proj_weight_mask.expand(3, embed_dim, embed_dim).reshape(3 * embed_dim, embed_dim)

        # expand out_proj_bias_mask from (embed_dim) to (3, embed_dim) by repeating it 3 times
        in_proj_bias_mask = out_proj_bias_mask.expand(3, embed_dim).reshape(3 * embed_dim)
        return in_proj_weight_mask, in_proj_bias_mask, out_proj_weight_mask, out_proj_bias_mask



def rename_parameter(obj, old_name, new_name):
    """
    function for renaming an instance parameter to another name.

    e.g.:
        linear = torch.nn.Linear(1,2)
        linear.weight.shape
        ---> torch.Size([2, 1])
        linear.weight_lin.shape
        ---> AttributeError

        rename_parameter(linear, "weight", "weight_lin")

        linear.weight_lin.shape
        ---> torch.Size([2, 1])
        linear.weight.shape
        ---> AttributeError

    Also works recursively for parameters of parameters, like:
        rename_parameter(linear, "weight.shape", "weight.shape_")

    Args:
        obj: the object where the attribute should be renamed
        old_name: the current name of the parameter
        new_name:the new name of the parameter

    Returns:
        None
    """
    def rename_param(obj, old_name, new_name):
        # print(obj.__dict__.get('_parameters').keys())
        obj.__dict__.get('_parameters')[new_name] = obj._parameters.pop(old_name)

    pre, _, post = old_name.rpartition('.')
    pren, _, postn = new_name.rpartition('.')
    return rename_param(rgetattr(obj, pre) if pre else obj, post, postn)


def rsetattr(obj, attr, val):
    """
    recursive setattr
    Args:
        obj: object
        attr: attribute name as string
        val: value to set the attribute to

    Returns:
        None
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
    recursive getattr
    Args:
        obj: object
        attr: attribute name as string
        *args:

    Returns:
        the attribute
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rdelattr(obj, attr, *args):
    """
    recursive delattr
    Args:
        obj: object
        attr: attribute name as string
        *args:

    Returns:

    """
    def _delattr(obj, attr):
        return delattr(obj, attr, *args)

    return functools.reduce(_delattr, [obj] + attr.split('.'))


def mask_class_wrapper(super_class, mode="linear", embedded_dims=64, k=7):
    """
    Function that creates a wrapper class that is a child class of the given super_class.
    Functionality:
        Every time the forward function of the created class is called, every parameter is renamed
        from <param_name> to <param_name> + "_". Then the masks are generated. Every parameter
        is multiplied by their corresponding mask. The result is stored in a parameter, which has
        the original name <param_name>. After that the forward function of the super_class is called.
        As we exchanged all parameters to the masked ones, this function also calculates with the
        masked ones.

        After the forward call of the super_class, all parameters are named back to their original name.

    Args:
        super_class: the super_class
        mode: the mask mode to use ("linear", "conv" or "mha_linear")
        embedded_dims: embedded_dims (only needed for mode="mha_linear")
        k: the exponent for the sigmoid function for mask creation f(x) = 1 / (1 + e^(-k * (x - b))), default = 7

    Returns:
        the created wrapper class that has prunable masks
    """

    def init_masks(self):
        """
        initialize the masks
        Args:
            self:

        Returns:
            None
        """

        self.masks = {}
        # iterate over all parameters of module
        for i, name in enumerate(self.names):
            if "bn." in name:
                continue
            # If current name is weight and the next name in the list is not called bias
            # -> weight with no bias
            if "weight" in name.split(".")[-1] and not (
                    i < len(self.names) - 1 and self.names[i + 1].split(".")[-1] == "bias"):

                # create mask for Conv2d or ConvModule
                if mode == "conv":
                    size_weight = rgetattr(self, name).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableMaskConv2d(feature_input, feature_output, kernel_size, k=k)

                    # for ConvModule
                    if i < len(self.names) - 2 and "bn.bias" in self.names[i + 2]:
                        self.masks[self.names[i + 2]] = self.masks[name]

                # create mask for MHA layer
                # the order of the parameters in MHA without bias  is (in_proj_weight, out_proj.weight)
                elif mode == "mha_linear" and "out_proj.weight" in name:
                    size_weight = rgetattr(self, name).size()
                    size_bias = size_weight[0]
                    self.masks[name] = LearnableMaskMHALinear(size_weight, int(size_bias), int(size_bias//embedded_dims), k=k)
                    self.masks["in_proj_weight"] = self.masks[name]

            # If current name is bias and the previous name in the list was weight
            # -> weight and bias belong together
            elif "bias" in name.split(".")[-1] and i != 0 and "weight" in self.names[i - 1].split(".")[
                -1]:
                if mode == "linear":
                    size_bias = rgetattr(self, name).size()
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    self.masks[name] = LearnableMaskLinear(size_weight, size_bias, k=k)
                    self.masks[self.names[i - 1]] = self.masks[name]
                elif mode == "conv":
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableMaskConv2d(feature_input, feature_output, kernel_size, k=k)
                    self.masks[self.names[i - 1]] = self.masks[name]
                elif mode == "mha_linear":
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    size_bias = size_weight[0]

                    if "out_proj.weight" in self.names[i - 1]:
                        self.masks[name] = LearnableMaskMHALinear(size_weight, int(size_bias), int(size_bias//embedded_dims), k=k)
                        self.masks["in_proj_weight"] = self.masks[name]
                        self.masks[self.names[i - 1]] = self.masks[name]
                        self.masks["in_proj_bias"] = self.masks[name]
                else:
                    raise NotImplementedError(f"Mode {mode} not implemented yet")

        self.module_list = nn.ModuleList(self.masks.values())
    def delete_masks(self):
        """
        delete masks
        Args:
            self:

        Returns:

        """
        for mask in self.masks.values():
            del mask
        del self.module_list
        self.masks = {}

    def reinit_masks(self):
        """
        reinit all maks by calling the reinit function from all masks
        Args:
            self:

        Returns:

        """
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
        """
        constructor function (__init__)
        calls super constructor and then initializes teh masks
        Args:
            self:
            *args:
            **kw:

        Returns:

        """
        super_class.__init__(self, *args, **kw)
        self.mask_class_wrapper = True
        self.names = [name for name, _ in self.named_parameters()]
        self.init_masks()
        self.training = True

    def reset_parameters_wrapper(self, *args, **kw):
        """
        reset parameters
        Args:
            self:
            *args:
            **kw:

        Returns:

        """
        return super_class._reset_parameters(self, *args, **kw)

    def get_mask_loss(self):
        """
        collects and sums up all partial loss terms from the masks
        Args:
            self:

        Returns:

        """
        loss = 0
        for mask in self.masks.values():
            loss = loss + mask.get_loss()
        return loss

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """
        wrapper for state_dict function of super_class

        sets the model in state "train" before outputting the state_dict
        and then restores the old model state.
        Args:
            self:
            *args:
            destination:
            prefix:
            keep_vars:

        Returns:

        """
        mode_before = self.training
        self.train()
        res = super_class.state_dict(self, *args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.train(mode_before)
        return res

    def train_wrapper(self, mode_train: bool = True, *args, **kw):
        """
        train wrapper

        During inference (mode_train=False ) it is unnecessary to compute the mask every time new.
        This would just increase the inference time. So this function detects, when the model is changed from
        a) (mode=train -> mode=eval) and b) (mode=eval -> mode=train).

        a) For case a: The masks for every parameter are computed and every parameter is multipled by the mask
        b) For case a: The masks for every parameter are computed and every parameter is divided by the mask



        Args:
            self:
            mode_train:
            *args:
            **kw:

        Returns:

        """
        has_changed = self.training != mode_train
        super_class.train(self, mode=mode_train, *args, **kw)
        if has_changed:
            with torch.no_grad():
                for i, name in enumerate(self.names):
                    if name in self.masks:
                        weight_idx, bias_idx = 0, 1
                        if mode == "mha_linear":
                            weight_idx, bias_idx = (0, 1) if "in_proj" in name else (2, 3)

                        mask = self.masks[name].get_mask()[weight_idx] if "weight" in name else self.masks[name].get_mask()[bias_idx]
                        if mode_train:
                            mask = torch.where(mask < 0.0001, 0.0, 1.0/mask)
                        else:
                            mask = torch.where(mask < 0.0001, 0.0, mask)
                        param = rgetattr(self, name)
                        if param.size() != mask.size():
                            print("err")
                            mask = self.masks.pop(name)
                            self.names.pop(i)
                            del mask
                        else:
                            param.copy_(param * mask)
        return self

    def initialize_permutation_if_empty(self, name, index):
        """
        initialize permutations for every mask if not already done
        Args:
            self:
            name:
            index:

        Returns:

        """
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
                                                    torch.zeros((rgetattr(self, name + "_").size()[0]), device=self.masks[name].get_device()))

    def forward_wrapper(self, *args, **kw):
        """
        forward wrapper

        Every time the forward function of the created class is called, every parameter is renamed
        from <param_name> to <param_name> + "_". Then the masks are generated. Every parameter
        is multiplied by their corresponding mask. The result is stored in a parameter, which has
        the original name <param_name>. After that the forward function of the super_class is called.
        As we exchanged all parameters to the masked ones, this function also calculates with the
        masked ones.

        After the forward call of the super_class, all parameters are named back to their original name.

        Args:
            self:
            *args:
            **kw:

        Returns:

        """
        if self.training:
            last_mask_obj = None
            last_mask = None
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

                    if self.masks[name] == last_mask_obj:
                        current_mask = last_mask
                    else:
                        current_mask = self.masks[name].get_mask()
                    mask = current_mask[weight_idx] if "weight" in name else current_mask[bias_idx]
                    last_mask_obj = self.masks[name]
                    last_mask = current_mask

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
        "reinit_masks": reinit_masks,
        "state_dict": state_dict
    })


def get_p1_values(module):
    """
    get all p1 parameters of all masks
    Args:
        module:

    Returns:

    """
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            p1_list[n] = p.p1
    return p1_list


def get_num_pruned(module):
    """
    get number of elements that could be pruned
    Args:
        module:

    Returns:

    """
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            p1_list[n] = (p.p1 * p.lr_mult_factor).int().float()
    return p1_list


def get_percentage_pruned(module):
    """
    get percentage of elements that could be pruned
    Args:
        module:

    Returns:

    """
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            p1_list[n] = (p.p1 * p.lr_mult_factor).int().float() / p.pruning_size
    return p1_list


def get_p1_loss(module):
    """
    get the sum of all mask losses from the model
    Args:
        module:

    Returns:

    """
    loss_sum = 0
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            loss_sum = loss_sum + p.get_loss()
    return loss_sum
