from typing import Optional, Tuple, Union
from ...tensor import Tensor
from ..function import Function


class Sum(Function):
    """
    Implements the sum operation with support for automatic differentiation.

    This operation computes the sum of elements along a specified dimension
    and calculates gradients for the input tensor during the backward pass.
    """

    @staticmethod
    def forward(
        ctx: Function, input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> Tensor:
        """
        Computes the forward pass for the sum operation.

        Args:
            ctx (Function): The context object for storing information for the backward pass.
            input (Tensor): The input tensor.
            dim (Optional[Union[int, Tuple[int, ...]]]): The dimension(s) along which to sum. If None, sums all elements.
            keepdim (bool): Whether to retain reduced dimensions in the output.

        Returns:
            Tensor: The output tensor containing the sum.
        """
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = input.data.shape

        output_data = input.data.sum(dim=dim, keepdim=keepdim)
        output = Tensor(output_data, requires_grad=input.requires_grad)

        if input.requires_grad:
            output.grad_fn = Sum(ctx)
            output.grad_fn.inputs = (input, dim, keepdim)

        return output

    def backward(self, grad_output: Tensor) -> Tuple[Tensor, Optional[None], Optional[None]]:
        """
        Computes the backward pass for the sum operation.

        Args:
            grad_output (Tensor): The gradient of the loss with respect to the output tensor.

        Returns:
            Tuple[Tensor, Optional[None], Optional[None]]: The gradient of the loss with respect to the input tensor,
            and `None` for the `dim` and `keepdim` arguments as they are not differentiable.
        """
        _, dim, keepdim = self.inputs

        if dim is None:
            grad_input_data = grad_output.data.expand(self.ctx.input_shape)
        else:
            grad_output_data = grad_output.data
            if not keepdim:
                dims = (dim,) if isinstance(dim, int) else dim
                for dim_idx in sorted(dims):
                    grad_output_data = grad_output_data.unsqueeze(dim_idx)

            grad_input_data = grad_output_data.expand(self.ctx.input_shape)

        grad_input = Tensor(grad_input_data)
        return grad_input, None, None
