from typing import Optional, Tuple
from ...tensor import Tensor
from ..function import Function


class Mul(Function):
    """
    Implements the element-wise multiplication operation with support for automatic differentiation.

    This operation computes the product of two tensors element-wise during the forward pass
    and calculates the corresponding gradients during the backward pass.
    """

    @staticmethod
    def forward(ctx: Function, a: Tensor, b: Tensor) -> Tensor:
        """
        Computes the forward pass for element-wise multiplication.

        Args:
            ctx (Function): The context object for storing information for the backward pass.
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
            Tensor: The output tensor resulting from the element-wise multiplication of `a` and `b`.
        """
        ctx.save_for_backward(a, b)
        output = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
        return output

    def backward(self, grad_output: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Computes the backward pass for element-wise multiplication.

        Args:
            grad_output (Optional[Tensor]): The gradient of the loss with respect to the output tensor.

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor]]: Gradients with respect to `a` and `b`.
        """
        a, b = self.ctx.saved_tensors
        grad_a = Tensor(b.data * grad_output.data) if a.requires_grad else None
        grad_b = Tensor(a.data * grad_output.data) if b.requires_grad else None
        return grad_a, grad_b
