# mytorch/autograd/functions/matmul.py

from ...tensor import Tensor
from ..function import Function


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        output = Tensor(a.data.mm(b.data), requires_grad=a.requires_grad or b.requires_grad)
        return output

    def backward(self, grad_output=None):
        a, b = self.ctx.saved_tensors
        grad_a = grad_output.mm(b.data.t()) if a.requires_grad else None
        grad_b = a.data.t().mm(grad_output) if b.requires_grad else None
        return grad_a, grad_b
