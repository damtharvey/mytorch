from mytorch.tensor import Tensor
from ..module import Module
from ...autograd.functions.relu import ReLU as ReLUFunction


class ReLU(Module):
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    ReLU is defined as:
        ReLU(x) = max(0, x)

    This is a commonly used activation function in neural networks to introduce non-linearity.

    Inherits:
        Module: Base class for all neural network modules.
    """

    def __init__(self) -> None:
        """
        Initializes the ReLU module.
        """
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Applies the ReLU activation function to the input tensor.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: A tensor with ReLU applied element-wise.
        """
        return ReLUFunction.apply(input)
