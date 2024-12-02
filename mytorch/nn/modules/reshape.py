# mytorch/nn/modules/reshape.py

from ..module import Module


class Reshape(Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.reshape(*self.shape)
