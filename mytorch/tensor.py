# mytorch/tensor.py

import torch


class Tensor:
    def __init__(self, data, requires_grad=False, is_leaf=True, grad_fn=None):
        if isinstance(data, Tensor):
            data = data.data
        self.data = torch.tensor(data, requires_grad=False)
        self.requires_grad = requires_grad
        self.grad = None
        self.is_leaf = is_leaf
        self.grad_fn = grad_fn  # The Function that created this Tensor

    def backward(self, grad_output=None):
        if not self.requires_grad:
            return

        if grad_output is None:
            if self.data.numel() != 1:
                raise RuntimeError("grad_output not specified for non-scalar tensor")
            grad_output = Tensor(torch.ones_like(self.data))

        if self.grad is None:
            self.grad = grad_output.data.clone()
        else:
            self.grad += grad_output.data

        if self.grad_fn:
            grad_inputs = self.grad_fn.backward(grad_output)
            for input_tensor, grad_input in zip(self.grad_fn.inputs, grad_inputs):
                if input_tensor.requires_grad:
                    input_tensor.backward(grad_input)

    # Define basic operations using magic methods
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.add import Add  # Local import

        return Add.apply(self, other)

    def mm(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.matmul import MatMul  # Local import

        return MatMul.apply(self, other)

    def relu(self):
        from .autograd.functions.relu import ReLU  # Local import

        return ReLU.apply(self)
