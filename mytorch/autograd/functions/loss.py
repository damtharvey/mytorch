from typing import Tuple, Optional
from ...tensor import Tensor
from ..function import Function
from .logsoftmax import LogSoftmax
import torch


class NLLLoss(Function):
    """
    Negative Log-Likelihood (NLL) Loss.

    This loss is typically used in combination with `LogSoftmax` for classification tasks.
    It computes the negative log-likelihood of the true class probabilities.
    """

    @staticmethod
    def forward(ctx: Function, input: Tensor, target: Tensor) -> Tensor:
        """
        Computes the forward pass for NLL loss.

        Args:
            ctx (Function): The context object for storing information for the backward pass.
            input (Tensor): The input tensor of shape `(N, C)`, where `N` is the batch size and `C` is the number of classes.
            target (Tensor): The ground truth labels of shape `(N,)`.

        Returns:
            Tensor: The scalar loss value.
        """
        batch_size = input.data.size(0)
        ctx.batch_size = batch_size
        ctx.input_shape = input.data.shape
        ctx.target = target

        loss_data = -input.data[range(batch_size), target].mean()
        loss = Tensor(loss_data, requires_grad=True)
        if input.requires_grad:
            loss.grad_fn = NLLLoss(ctx)
            loss.grad_fn.inputs = (input, target)

        return loss

    def backward(self, grad_output: Optional[Tensor] = None) -> Tuple[Tensor, Optional[None]]:
        """
        Computes the backward pass for NLL loss.

        Args:
            grad_output (Optional[Tensor]): The gradient of the loss with respect to the output.

        Returns:
            Tuple[Tensor, Optional[None]]: Gradients with respect to the input tensor and `None` for the target.
        """
        if grad_output is None:
            grad_output = Tensor(1.0)

        input, target = self.inputs
        batch_size = self.ctx.batch_size

        grad_input_data = torch.zeros_like(input.data)
        grad_input_data[range(batch_size), target] = -1.0 / batch_size
        grad_input = Tensor(grad_input_data) * grad_output.data

        return grad_input, None


class CrossEntropyLoss(Function):
    """
    Cross-Entropy Loss.

    This loss combines `LogSoftmax` and `NLLLoss` into a single operation.
    It is numerically stable and commonly used in classification tasks.
    """

    @staticmethod
    def forward(ctx: Function, input: Tensor, target: Tensor) -> Tensor:
        """
        Computes the forward pass for Cross-Entropy loss.

        Args:
            ctx (Function): The context object for storing information for the backward pass.
            input (Tensor): The input tensor of shape `(N, C)` or `(C,)` for 1D input.
            target (Tensor): The ground truth labels of shape `(N,)`.

        Returns:
            Tensor: The scalar loss value.
        """
        is_1d = input.data.dim() == 1
        if is_1d:
            input = Tensor(input.data.unsqueeze(0), requires_grad=input.requires_grad)

        log_softmax = LogSoftmax.apply(input, 1)
        loss = NLLLoss.apply(log_softmax, target)

        ctx.save_for_backward(log_softmax, target, is_1d)

        return loss if not is_1d else Tensor(loss.data.squeeze(0), requires_grad=loss.requires_grad)

    def backward(self, grad_output: Optional[Tensor] = None) -> Tuple[Tensor, Optional[None]]:
        """
        Computes the backward pass for Cross-Entropy loss.

        Args:
            grad_output (Optional[Tensor]): The gradient of the loss with respect to the output.

        Returns:
            Tuple[Tensor, Optional[None]]: Gradients with respect to the input tensor and `None` for the target.
        """
        if grad_output is None:
            grad_output = Tensor(1.0)

        log_softmax, target, is_1d = self.ctx.saved_tensors
        batch_size = log_softmax.data.size(0)

        softmax = log_softmax.data.exp()
        grad_input_data = softmax
        grad_input_data[range(batch_size), target] -= 1
        grad_input_data /= batch_size

        grad_input = Tensor(grad_input_data) * grad_output.data

        # Squeeze gradient if input was 1D
        return grad_input if not is_1d else Tensor(grad_input.data.squeeze(0)), None
