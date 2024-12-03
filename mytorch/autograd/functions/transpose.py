# mytorch/autograd/functions/transpose.py

from ...tensor import Tensor
from ..function import Function


class Transpose(Function):
    @staticmethod
    def forward(ctx, input, dim0, dim1):
        ctx.dim0 = dim0
        ctx.dim1 = dim1
        output_data = input.data.transpose(dim0, dim1)
        output = Tensor(output_data, requires_grad=input.requires_grad)
        if input.requires_grad:
            # Instantiate Transpose with ctx
            output.grad_fn = Transpose(ctx)
            output.grad_fn.inputs = (input, dim0, dim1)
        return output

    def backward(self, grad_output):
        input, dim0, dim1 = self.inputs
        # Gradient of transpose is transpose of gradient
        grad_input_data = grad_output.data.transpose(dim0, dim1)
        grad_input = Tensor(grad_input_data)
        return (grad_input, None, None)  # Gradients for input, dim0, dim1
