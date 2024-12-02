# mytorch/autograd/functions/matmul.py

from ...tensor import Tensor
from ..function import Function


class MatMul(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output_data = input.data.matmul(weight.data)
        output = Tensor(output_data, requires_grad=input.requires_grad or weight.requires_grad)
        if input.requires_grad or weight.requires_grad:
            output.grad_fn = MatMul(ctx)
            output.grad_fn.inputs = (input, weight)
        return output

    def backward(self, grad_output):
        input, weight = self.inputs
        grad_input = grad_output.data.matmul(weight.data.t())
        grad_weight = input.data.t().matmul(grad_output.data)
        grad_input_tensor = Tensor(grad_input)
        grad_weight_tensor = Tensor(grad_weight)
        return (grad_input_tensor, grad_weight_tensor)
