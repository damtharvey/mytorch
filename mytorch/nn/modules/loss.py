# mytorch/nn/modules/cross_entropy_loss.py

from ..module import Module
from ...autograd.functions.loss import CrossEntropyLoss as CrossEntropyLossFunction  # Ensure this import exists

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return CrossEntropyLossFunction.apply(input, target)
