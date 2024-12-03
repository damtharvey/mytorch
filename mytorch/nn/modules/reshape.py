# mytorch/nn/modules/reshape.py

from ..module import Module
from mytorch.autograd.functions import Reshape as ReshapeFunction


class Reshape(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, shape):
        return ReshapeFunction.apply(input, shape)
