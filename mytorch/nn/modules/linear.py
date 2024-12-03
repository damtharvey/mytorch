# mytorch/nn/modules/linear.py

import torch

from ...tensor import Tensor
from ..module import Module
from ...autograd.functions.matmul import MatMul
from ...autograd.functions.add import Add
from ...autograd.functions.transpose import Transpose
from ..init import kaiming_uniform


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._parameters["weight"] = Tensor(torch.empty(out_features, in_features), requires_grad=True)
        self._parameters["bias"] = Tensor(torch.zeros(out_features), requires_grad=True)

        kaiming_uniform(self._parameters["weight"])
        bound = 1 / in_features**0.5 if in_features > 0 else 0
        self._parameters["bias"].data.uniform_(-bound, bound)

    def forward(self, input):
        weight = self._parameters["weight"]
        transposed_weight = Transpose.apply(weight, 0, 1)  # (in_features, out_features)
        bias = self._parameters["bias"]
        out = MatMul.apply(input, transposed_weight)
        out = Add.apply(out, bias)
        return out

    @property
    def weight(self):
        return self._parameters["weight"]

    @property
    def bias(self):
        return self._parameters["bias"]
