import math
from typing import Optional, Tuple

import numpy as np
import torch
from mmengine.analysis import FlopAnalyzer
from mmengine.analysis.print_helper import _format_size
from torch import nn, functional, Tensor
from torch.nn import Parameter
from torch.nn.functional import _in_projection_packed, softmax, _mha_shape_check, linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


class MLP(nn.Module):
    def __init__(self, input_features, output_features):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        return out


class DGLLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, top_r=0.5) -> None:
        super().__init__()
        self.out_features = out_features

        self.layer_norm = nn.LayerNorm(in_features)
        self.top_k = int(out_features * top_r)
        self.gate_predictor = MLP(out_features, out_features)

    def set_topr(self, top_r):
        assert 0.0 <= top_r <= 1.0, f"the topr value must be between 0 and 1, but got {top_r}"
        self.top_k = int(self.out_features * top_r)

    def forward(self, x, l_w, l_b=None):
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(1)

        x = x.permute(1, 0, 2)
        n, l, c = x.size()
        n = int(n)
        l = int(l)
        c = int(c)
        x_avg = torch.squeeze(torch.nn.functional.avg_pool1d(x.permute(0, 2, 1), l, l, 0, False, True), 2)
        # x_avg has now the size: (n, c)
        x_norm = self.layer_norm(x_avg)

        gate_predictor_input = torch.nn.functional.linear(x_norm, l_w, l_b)
        gate_predictor_logits = self.gate_predictor(gate_predictor_input)

        mask = torch.zeros((n, self.out_features), device=x.get_device())
        output = torch.zeros((n, self.out_features, l), device=x.get_device())
        indices = torch.topk(gate_predictor_logits, k=self.top_k, dim=1).indices
        w_list = []
        for i_n in range(0, n):  # TODO: Implement without for loop
            mask[i_n, indices[i_n]] = 1
            w_list.append(l_w[indices[i_n], :])
        w = torch.stack(w_list)
        w = w.permute(0, 2, 1).unsqueeze(1)
        x_p = x.unsqueeze(2)
        output_not_null = torch.matmul(x_p, w).squeeze(2).permute(0, 2, 1)

        bias = l_b.unsqueeze(1).repeat(1, l)
        for i_n in range(0, n):
            output[i_n, indices[i_n]] = output_not_null[i_n, :] + (bias[indices[i_n]] if l_b is not None else 0)

        return output.permute(2, 0, 1)


class DynamicGatedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.bias = True

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = False
        self.head_dim = embed_dim // num_heads

        self.dgl_layer_q = DGLLayer(embed_dim, embed_dim, top_r=1.0)
        self.dgl_layer_k = DGLLayer(embed_dim, embed_dim, top_r=1.0)
        self.dgl_layer_v = DGLLayer(embed_dim, embed_dim, top_r=1.0)

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if self.bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))

        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=self.bias, **factory_kwargs)
        self.dgl_layer_out = DGLLayer(embed_dim, embed_dim, top_r=1.0)
        self._reset_parameters()

    def set_topr_dgl(self, topr):
        assert 0.0 <= topr <= 1.0, f"the topr value must be between 0 and 1, but got {topr}"
        self.dgl_layer_q.set_topr(topr)
        self.dgl_layer_k.set_topr(topr)
        self.dgl_layer_v.set_topr(topr)
        self.dgl_layer_out.set_topr(topr)

    def _inweight_projection(self, q, k, v, w, b):
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return self.dgl_layer_q(q, w_q, b_q), self.dgl_layer_k(k, w_k, b_k), self.dgl_layer_v(v, w_v, b_v)

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                average_attn_weights=True) -> Tuple[Tensor, Optional[Tensor]]:
        #print(key.size(), query.dim(), self.batch_first)
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        is_batched = _mha_shape_check(query, key, value, None, None, self.num_heads)

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q, k, v = self._inweight_projection(query, key, value, self.in_proj_weight, self.in_proj_bias)

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        #print(k.size(), k.shape[0], bsz, self.num_heads, self.head_dim)
        k = k.view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)

        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = self.dgl_layer_out(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(2))

        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class DynamicGatedConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        assert self.kernel_size[0] == 1 and self.kernel_size[
            1] == 1, f"the kernel size must be 1x1 for DynamicGatedConv2d, but got {self.kernel_size}"
        self.cnn_dgl = DGLLayer(in_channels, out_channels, top_r=0.5)

    def forward(self, input: Tensor) -> Tensor:
        is_batched = input.dim() == 4
        if not is_batched:
            input = input.unsqueeze(0)
        b, c, h, w = input.size()
        out = self.cnn_dgl(input.permute(2, 3, 0, 1).reshape(h * w, b, c), self.weight.squeeze(3).squeeze(2),
                           self.bias).permute(1, 2, 0).reshape(b, self.cnn_dgl.out_features, h, w)
        if not is_batched:
            out = out.squeeze(0)
        return out

    def forward_ungated(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

    def set_topr_dgl(self, topr):
        assert 0.0 <= topr <= 1.0, f"the topr value must be between 0 and 1, but got {topr}"
        self.cnn_dgl.set_topr(topr)
