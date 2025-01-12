from typing import Tuple, Optional
from ...tensor import Tensor
from ..function import Function


class LogSoftmax(Function):
    """
    Implements the LogSoftmax operation with support for automatic differentiation.

    This operation computes the logarithm of the softmax function along a specified dimension
    for numerical stability and is commonly used in classification tasks.
    """

    @staticmethod
    def forward(ctx: Function, input: Tensor, dim: int) -> Tensor:
        """
        Computes the forward pass for the LogSoftmax operation.

        Args:
            ctx (Function): The context object for storing information for the backward pass.
            input (Tensor): The input tensor.
            dim (int): The dimension along which to compute the LogSoftmax.

        Returns:
            Tensor: The output tensor containing LogSoftmax values along the specified dimension.
        """
        ctx.dim = dim

        # Compute numerically stable log-softmax
        max_input = input.data.max(dim=dim, keepdim=True)[0]  # Max for numerical stability
        input_stable = input.data - max_input
        sum_exp = input_stable.exp().sum(dim=dim, keepdim=True)
        log_softmax_data = input_stable - sum_exp.log()

        output = Tensor(log_softmax_data, requires_grad=input.requires_grad)
        if input.requires_grad:
            output.grad_fn = LogSoftmax(ctx)
            output.grad_fn.inputs = (input, dim)

        return output

    def backward(self, grad_output: Tensor) -> Tuple[Tensor, Optional[None]]:
        """
        Computes the backward pass for the LogSoftmax operation.

        Args:
            grad_output (Tensor): The gradient of the loss with respect to the output of LogSoftmax.

        Returns:
            Tuple[Tensor, Optional[None]]: The gradient of the loss with respect to the input tensor,
            and `None` for the `dim` argument since it is not differentiable.
        """
        input, dim = self.inputs

        # Compute softmax probabilities
        softmax = input.data.exp()

        # Gradient with respect to the input
        grad_input_data = grad_output.data - (grad_output.data.sum(dim=dim, keepdim=True) * softmax)
        grad_input = Tensor(grad_input_data)

        return grad_input, None  # No gradient for `dim`
