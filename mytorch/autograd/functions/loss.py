# mytorch/autograd/functions/loss.py

from ...tensor import Tensor
from ..function import Function
from .logsoftmax import LogSoftmax
import torch


class NLLLoss(Function):
    @staticmethod
    def forward(ctx, input, target):
        batch_size = input.data.size(0)
        ctx.batch_size = batch_size
        ctx.input_shape = input.data.shape
        ctx.target = target
        loss_data = -input.data[range(batch_size), target].mean()
        loss = Tensor(loss_data, requires_grad=True)
        if input.requires_grad:
            # Instantiate NLLLoss with ctx
            loss.grad_fn = NLLLoss(ctx)
            loss.grad_fn.inputs = (input, target)
        return loss

    def backward(self, grad_output):
        if grad_output is None:
            grad_output = Tensor(1.0)

        input, target = self.inputs
        batch_size = self.ctx.batch_size
        grad_input_data = torch.zeros_like(input.data)
        grad_input_data[range(batch_size), target] = -1.0 / batch_size
        grad_input = Tensor(grad_input_data) * grad_output.data
        return (grad_input, None)  # Corresponding to (input, target)


class CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, input, target):
        log_softmax = LogSoftmax.apply(input, 1)
        loss = NLLLoss.apply(log_softmax, target)
        ctx.save_for_backward(log_softmax, target)
        return loss

    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = Tensor(1.0)

        log_softmax, target = self.ctx.saved_tensors
        batch_size = log_softmax.data.size(0)
        softmax = log_softmax.data.exp()
        grad_input_data = softmax
        grad_input_data[range(batch_size), target] -= 1
        grad_input_data /= batch_size
        grad_input = Tensor(grad_input_data) * grad_output.data
        return (grad_input, None)  # Corresponding to (input, target)
