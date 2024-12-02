# mytorch/autograd/functions/relu.py

from ...tensor import Tensor
from ..function import Function

class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output_data = input.data.clamp(min=0)
        output = Tensor(output_data, requires_grad=input.requires_grad)
        if input.requires_grad:
            output.grad_fn = ReLU(ctx)
            output.grad_fn.inputs = (input,)
        return output

    def backward(self, grad_output):
        input, = self.inputs
        grad_input_data = grad_output.data.clone()
        grad_input_data[input.data <= 0] = 0
        grad_input = Tensor(grad_input_data)
        return (grad_input,)
