# mytorch/autograd/functions/add.py

from ...tensor import Tensor
from ..function import Function


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        output = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
        return output

    def backward(self, grad_output=None):
        a, b = self.ctx.saved_tensors
        grad_a = grad_output if a.requires_grad else None
        grad_b = grad_output if b.requires_grad else None
        return grad_a, grad_b
