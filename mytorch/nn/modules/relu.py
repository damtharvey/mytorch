# mytorch/nn/modules/relu.py

from ...tensor import Tensor
from ..module import Module


class ReLU(Module):
    def forward(self, input):
        return input.relu()
