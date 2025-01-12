from typing import Tuple
from ..module import Module
from mytorch.tensor import Tensor
from mytorch.autograd.functional import Reshape as ReshapeFunction


class Reshape(Module):
    """
    A module that reshapes an input tensor to a specified shape.

    This module provides a simple way to apply the `Reshape` operation within a model.
    """

    def __init__(self) -> None:
        """
        Initializes the Reshape module.
        """
        super().__init__()

    def forward(self, input: Tensor, shape: Tuple[int, ...]) -> Tensor:
        """
        Reshapes the input tensor to the specified shape.

        Args:
            input (Tensor): The input tensor to reshape.
            shape (Tuple[int, ...]): The target shape for the tensor.

        Returns:
            Tensor: The reshaped tensor.
        """
        return ReshapeFunction.apply(input, shape)
