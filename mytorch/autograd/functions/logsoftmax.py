# mytorch/autograd/functions/logsoftmax.py

from ...tensor import Tensor
from ..function import Function


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.dim = dim
        max_input = input.data.max(dim=dim, keepdim=True)[0]
        input_stable = input.data - max_input
        sum_exp = input_stable.exp().sum(dim=dim, keepdim=True)
        log_softmax_data = input_stable - sum_exp.log()
        output = Tensor(log_softmax_data, requires_grad=input.requires_grad)
        if input.requires_grad:
            output.grad_fn = LogSoftmax(ctx)
            output.grad_fn.inputs = (input, dim)
        return output

    def backward(self, grad_output):
        input, dim = self.inputs
        softmax = input.data.exp()
        grad_input_data = grad_output.data - (grad_output.data.sum(dim=dim, keepdim=True) * softmax)
        grad_input = Tensor(grad_input_data)
        return (grad_input, None)  # Corresponding to (input, dim)
