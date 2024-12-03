# mytorch/autograd/functions/reshape.py

from ...tensor import Tensor
from ..function import Function


class Reshape(Function):
    @staticmethod
    def forward(ctx, input, shape):
        ctx.original_shape = input.data.shape
        output_data = input.data.view(*shape)
        output = Tensor(output_data, requires_grad=input.requires_grad)
        if input.requires_grad:
            # Instantiate Reshape with ctx
            output.grad_fn = Reshape(ctx)
            output.grad_fn.inputs = (input, shape)
        return output

    def backward(self, grad_output):
        input, shape = self.inputs
        grad_input_data = grad_output.data.view(self.ctx.original_shape)
        grad_input = Tensor(grad_input_data)
        return (grad_input, None)  # Corresponding to (input, shape)
