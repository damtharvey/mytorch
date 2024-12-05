from ..module import Module
from mytorch.tensor import Tensor
from ...autograd.functions.transpose import Transpose


class TransposeModule(Module):
    """
    A module that transposes the specified dimensions of an input tensor.

    Transposition swaps two dimensions of the input tensor, which can be useful
    in tasks such as changing the layout of data or preparing tensors for certain operations.
    """

    def __init__(self, dim0: int, dim1: int) -> None:
        """
        Initializes the Transpose module.

        Args:
            dim0 (int): The first dimension to transpose.
            dim1 (int): The second dimension to transpose.
        """
        super().__init__()
        self.dim0: int = dim0
        self.dim1: int = dim1

    def forward(self, input: Tensor) -> Tensor:
        """
        Applies the transpose operation to the input tensor.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The transposed tensor with `dim0` and `dim1` swapped.
        """
        return Transpose.apply(input, self.dim0, self.dim1)
