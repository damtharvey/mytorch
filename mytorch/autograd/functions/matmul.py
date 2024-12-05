from typing import Tuple
from ...tensor import Tensor
from ..function import Function


class MatMul(Function):
    """
    Implements the matrix multiplication operation with support for automatic differentiation.

    This operation performs a matrix multiplication during the forward pass
    and computes the corresponding gradients during the backward pass.
    """

    @staticmethod
    def forward(ctx: Function, input: Tensor, weight: Tensor) -> Tensor:
        """
        Computes the forward pass for the matrix multiplication operation.

        Args:
            ctx (Function): The context object for storing information for the backward pass.
            input (Tensor): The input tensor of shape `(N, M)`.
            weight (Tensor): The weight tensor of shape `(M, P)`.

        Returns:
            Tensor: The output tensor of shape `(N, P)` resulting from the matrix multiplication.
        """
        ctx.save_for_backward(input, weight)

        output_data = input.data.matmul(weight.data)
        output = Tensor(output_data, requires_grad=input.requires_grad or weight.requires_grad)
        if input.requires_grad or weight.requires_grad:
            output.grad_fn = MatMul(ctx)
            output.grad_fn.inputs = (input, weight)

        return output

    def backward(self, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the backward pass for the matrix multiplication operation.

        Args:
            grad_output (Tensor): The gradient of the loss with respect to the output of this operation.

        Returns:
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to `input` and `weight`.
        """
        input, weight = self.inputs

        grad_input_data = grad_output.data.matmul(weight.data.t())
        grad_weight_data = input.data.t().matmul(grad_output.data)
        grad_input = Tensor(grad_input_data)
        grad_weight = Tensor(grad_weight_data)

        return grad_input, grad_weight
