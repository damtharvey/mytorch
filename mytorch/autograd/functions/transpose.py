from typing import Tuple, Optional
from ...tensor import Tensor
from ..function import Function


class Transpose(Function):
    """
    Implements the transpose operation with support for automatic differentiation.

    This operation swaps two specified dimensions of a tensor.
    """

    @staticmethod
    def forward(ctx: Function, input: Tensor, dim0: int, dim1: int) -> Tensor:
        """
        Computes the forward pass for the transpose operation.

        Args:
            ctx (Function): The context object for storing information for the backward pass.
            input (Tensor): The input tensor.
            dim0 (int): The first dimension to transpose.
            dim1 (int): The second dimension to transpose.

        Returns:
            Tensor: The transposed tensor with `dim0` and `dim1` swapped.
        """
        ctx.dim0 = dim0
        ctx.dim1 = dim1
        output_data = input.data.transpose(dim0, dim1)
        output = Tensor(output_data, requires_grad=input.requires_grad)
        if input.requires_grad:
            output.grad_fn = Transpose(ctx)
            output.grad_fn.inputs = (input, dim0, dim1)
        return output

    def backward(self, grad_output: Tensor) -> Tuple[Tensor, Optional[None], Optional[None]]:
        """
        Computes the backward pass for the transpose operation.

        Args:
            grad_output (Tensor): The gradient of the loss with respect to the output tensor.

        Returns:
            Tuple[Tensor, Optional[None], Optional[None]]: The gradient of the loss with respect to the input tensor,
            and `None` for `dim0` and `dim1` as they are not differentiable.
        """
        input, dim0, dim1 = self.inputs
        grad_input_data = grad_output.data.transpose(dim0, dim1)
        grad_input = Tensor(grad_input_data)
        return grad_input, None, None
