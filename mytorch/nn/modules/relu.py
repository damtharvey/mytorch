# mytorch/nn/modules/relu.py

from ..module import Module
from ...tensor import Tensor
from ...autograd.functions.relu import ReLU as ReLUFunction  # Ensure this import exists

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return  ReLUFunction.apply(input)
