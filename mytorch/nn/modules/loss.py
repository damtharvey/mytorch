# mytorch/nn/modules/loss.py

from ..module import Module
from ...autograd.functions.loss import CrossEntropyLoss as CrossEntropyLossFunction


class CrossEntropyLoss(Module):
    def forward(self, input, target):
        return CrossEntropyLossFunction.apply(input, target)
