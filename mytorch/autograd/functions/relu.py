# mytorch/autograd/functions/relu.py

from ...tensor import Tensor
from ..function import Function


class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = Tensor(input.data.clamp(min=0), requires_grad=input.requires_grad)
        return output

    def backward(self, grad_output=None):
        (input,) = self.ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.data[input.data < 0] = 0
        return grad_input
