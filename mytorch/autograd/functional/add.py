from typing import Tuple
from ...tensor import Tensor
from ..function import Function


class Add(Function):
    """
    Implements the addition operation for tensors with support for automatic differentiation.

    This operation performs element-wise addition of two tensors during the forward pass
    and computes the corresponding gradients during the backward pass.
    """

    @staticmethod
    def forward(ctx: Function, input: Tensor, bias: Tensor) -> Tensor:
        """
        Computes the forward pass for the addition operation.

        Args:
            ctx (Function): The context object for storing information for the backward pass.
            input (Tensor): The first input tensor.
            bias (Tensor): The second input tensor (e.g., bias).

        Returns:
            Tensor: The result of adding `input` and `bias` element-wise.
        """
        ctx.save_for_backward(input, bias)
        output_data = input.data + bias.data
        output = Tensor(output_data, requires_grad=input.requires_grad or bias.requires_grad)
        if input.requires_grad or bias.requires_grad:
            output.grad_fn = Add(ctx)
            output.grad_fn.inputs = (input, bias)
        return output

    def backward(self, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the backward pass for the addition operation.

        Args:
            grad_output (Tensor): The gradient of the loss with respect to the output tensor.

        Returns:
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to `input` and `bias`.
        """
        grad_input = grad_output.data
        grad_bias = grad_output.data.sum(dim=0)
        grad_input_tensor = Tensor(grad_input)
        grad_bias_tensor = Tensor(grad_bias)
        return grad_input_tensor, grad_bias_tensor
