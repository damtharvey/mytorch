from mytorch.tensor import Tensor
from ..module import Module
from ...autograd.functional.loss import CrossEntropyLoss as CrossEntropyLossFunction


class CrossEntropyLoss(Module):
    """
    Computes the cross-entropy loss between predictions and target labels.

    This loss combines `LogSoftmax` and `Negative Log Likelihood (NLL)` in a single operation,
    which is more numerically stable than computing them separately.

    Inherits:
        Module: Base class for all neural network modules.
    """

    def __init__(self) -> None:
        """
        Initializes the CrossEntropyLoss module.
        """
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Computes the forward pass of the cross-entropy loss.

        Args:
            input (Tensor): The predicted logits of shape `(N, C)`, where `N` is the batch size and `C` is the number of classes.
            target (Tensor): The ground truth labels of shape `(N,)`, where each value is an index in the range `[0, C-1]`.

        Returns:
            Tensor: The computed cross-entropy loss.
        """
        return CrossEntropyLossFunction.apply(input, target)
