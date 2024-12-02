# mytorch/autograd/functions/logsoftmax.py

from ...tensor import Tensor
from ..function import Function
import torch


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input, dim):
        max_input = input.data.max(dim=dim, keepdim=True)[0]
        input_stable = input.data - max_input  # For numerical stability
        sum_exp = input_stable.exp().sum(dim=dim, keepdim=True)
        log_softmax_data = input_stable - sum_exp.log()
        output = Tensor(log_softmax_data, requires_grad=input.requires_grad)

        ctx.save_for_backward(output, dim)
        return output

    def backward(self, grad_output=None):
        output, dim = self.ctx.saved_tensors
        softmax = output.data.exp()
        grad_input_data = grad_output.data - (grad_output.data.sum(dim=dim, keepdim=True) * softmax)
        grad_input = Tensor(grad_input_data)
        return grad_input, None  # None for 'dim' since it's not a tensor
