# mytorch/autograd/functions/sum.py

from ...tensor import Tensor
from ..function import Function


class Sum(Function):
    @staticmethod
    def forward(ctx, input, dim=None, keepdim=False):
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = input.data.shape
        output_data = input.data.sum(dim=dim, keepdim=keepdim)
        output = Tensor(output_data, requires_grad=input.requires_grad)
        if input.requires_grad:
            # Instantiate Sum with ctx
            output.grad_fn = Sum(ctx)
            output.grad_fn.inputs = (input, dim, keepdim)
        return output

    def backward(self, grad_output):
        input, dim, keepdim = self.inputs
        if dim is None:
            grad_input_data = grad_output.data.expand(self.ctx.input_shape)
        else:
            grad_output_data = grad_output.data
            if not keepdim:
                if isinstance(dim, int):
                    dims = (dim,)
                else:
                    dims = dim
                for dim_idx in sorted(dims):
                    grad_output_data = grad_output_data.unsqueeze(dim_idx)
            grad_input_data = grad_output_data.expand(self.ctx.input_shape)
        grad_input = Tensor(grad_input_data)
        return (grad_input, None, None)  # Corresponding to (input, dim, keepdim)
