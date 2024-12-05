from typing import Tuple, Optional
from ...tensor import Tensor
from ..function import Function


class Reshape(Function):
    """
    Implements the reshape operation with support for automatic differentiation.

    This operation changes the shape of the input tensor without changing its data.
    """

    @staticmethod
    def forward(ctx: Function, input: Tensor, shape: Tuple[int, ...]) -> Tensor:
        """
        Computes the forward pass for the reshape operation.

        Args:
            ctx (Function): The context object for storing information for the backward pass.
            input (Tensor): The input tensor.
            shape (Tuple[int, ...]): The target shape for the tensor.

        Returns:
            Tensor: The reshaped tensor.
        """
        ctx.original_shape = input.data.shape
        output_data = input.data.view(*shape)
        output = Tensor(output_data, requires_grad=input.requires_grad)
        if input.requires_grad:
            output.grad_fn = Reshape(ctx)
            output.grad_fn.inputs = (input, shape)
        return output

    def backward(self, grad_output: Tensor) -> Tuple[Tensor, Optional[None]]:
        """
        Computes the backward pass for the reshape operation.

        Args:
            grad_output (Tensor): The gradient of the loss with respect to the output tensor.

        Returns:
            Tuple[Tensor, Optional[None]]: The gradient of the loss with respect to the input tensor,
            and `None` for the `shape` argument since it is not differentiable.
        """
        grad_input_data = grad_output.data.view(self.ctx.original_shape)
        grad_input = Tensor(grad_input_data)
        return grad_input, None
