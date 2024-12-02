# mytorch/autograd/functions/add.py

from ...tensor import Tensor
from ..function import Function


class Add(Function):
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        output_data = input.data + bias.data
        output = Tensor(output_data, requires_grad=input.requires_grad or bias.requires_grad)
        if input.requires_grad or bias.requires_grad:
            output.grad_fn = Add(ctx)
            output.grad_fn.inputs = (input, bias)
        return output

    def backward(self, grad_output):
        # input, bias = self.inputs
        grad_input = grad_output.data
        grad_bias = grad_output.data.sum(dim=0)
        grad_input_tensor = Tensor(grad_input)
        grad_bias_tensor = Tensor(grad_bias)
        return (grad_input_tensor, grad_bias_tensor)
