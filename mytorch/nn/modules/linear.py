# mytorch/nn/modules/linear.py

from ...tensor import Tensor
from ..module import Module
from ...autograd.functions.matmul import MatMul
from ...autograd.functions.add import Add
from ...autograd.functions.transpose import Transpose
import torch
import math


def calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out cannot be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def kaiming_uniform(tensor, a=math.sqrt(5)):
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    bound = math.sqrt(6 / fan_in) if fan_in > 0 else 0
    tensor.data.uniform_(-bound, bound)



class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._parameters['weight'] = Tensor(torch.empty(out_features, in_features), requires_grad=True)
        self._parameters['bias'] = Tensor(torch.zeros(out_features), requires_grad=True)
        kaiming_uniform(self._parameters['weight'])
        # Initialize bias uniformly
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        self._parameters['bias'].data.uniform_(-bound, bound)

    def forward(self, input):
        weight = self._parameters['weight']
        transposed_weight = Transpose.apply(weight, 0, 1)  # (in_features, out_features)
        bias = self._parameters['bias']
        out = MatMul.apply(input, transposed_weight)
        out = Add.apply(out, bias)
        return out
    
    @property
    def weight(self):
        return self._parameters['weight']

    @property
    def bias(self):
        return self._parameters['bias']
