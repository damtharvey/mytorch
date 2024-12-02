# mytorch/autograd/functions/loss.py

from ...tensor import Tensor
from ..function import Function
from .logsoftmax import LogSoftmax
import torch


class NLLLoss(Function):
    @staticmethod
    def forward(ctx, input, target):
        batch_size = input.data.size(0)
        ctx.save_for_backward(input, target, batch_size)
        # Negative log-likelihood loss
        loss_data = -input.data[range(batch_size), target].mean()
        output = Tensor(loss_data, requires_grad=True)
        return output

    @staticmethod
    def backward(ctx, grad_output=None):
        if grad_output is None:
            grad_output = Tensor(1.0)
        input, target, batch_size = ctx.saved_tensors
        grad_input_data = torch.zeros_like(input.data)
        grad_input_data[range(batch_size), target] = -1.0 / batch_size
        grad_input = Tensor(grad_input_data)
        return grad_input, None  # None for 'target' as it doesn't require gradients


class CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, input, target):
        # Apply log-softmax
        log_softmax = LogSoftmax.apply(input, 1)
        # Compute NLLLoss
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
        return (grad_input, None)  # Return a tuple matching inputs
