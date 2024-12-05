from typing import Tuple
from ...tensor import Tensor
from ..function import Function


class ReLU(Function):
    """
    Implements the Rectified Linear Unit (ReLU) operation with support for automatic differentiation.

    ReLU is defined as:
        ReLU(x) = max(0, x)

    This operation introduces non-linearity into the network and zeroes out negative values.
    """

    @staticmethod
    def forward(ctx: Function, input: Tensor) -> Tensor:
        """
        Computes the forward pass for the ReLU operation.

        Args:
            ctx (Function): The context object for storing information for the backward pass.
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor with ReLU applied element-wise.
        """
        ctx.save_for_backward(input)
        output_data = input.data.clamp(min=0)
        output = Tensor(output_data, requires_grad=input.requires_grad)
        if input.requires_grad:
            output.grad_fn = ReLU(ctx)
            output.grad_fn.inputs = (input,)
        return output

    def backward(self, grad_output: Tensor) -> Tuple[Tensor]:
        """
        Computes the backward pass for the ReLU operation.

        Args:
            grad_output (Tensor): The gradient of the loss with respect to the output tensor.

        Returns:
            Tuple[Tensor]: The gradient of the loss with respect to the input tensor.
        """
        (input,) = self.inputs
        grad_input_data = grad_output.data.clone()
        grad_input_data[input.data <= 0] = 0
        grad_input = Tensor(grad_input_data)
        return (grad_input,)
