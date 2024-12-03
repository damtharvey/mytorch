# mytorch/nn/modules/transpose.py

from ..module import Module
from ...autograd.functions.transpose import Transpose


class TransposeModule(Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, input):
        return Transpose.apply(input, self.dim0, self.dim1)
