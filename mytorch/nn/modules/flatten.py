# mytorch/nn/modules/flatten.py

from mytorch.nn.module import Module
from mytorch.tensor import Tensor


class Flatten(Module):
    def forward(self, input):
        batch_size = input.data.shape[0]
        return Tensor(input.data.view(batch_size, -1), requires_grad=input.requires_grad)
