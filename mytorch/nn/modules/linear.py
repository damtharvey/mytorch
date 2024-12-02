# mytorch/nn/modules/linear.py

import torch
from ...tensor import Tensor
from ..module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Tensor(torch.randn(out_features, in_features), requires_grad=True)
        if bias:
            self.bias = Tensor(torch.zeros(out_features), requires_grad=True)
        else:
            self.bias = None

    def forward(self, input):
        output = input.mm(self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output
