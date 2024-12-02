# mytorch/tensor.py

import torch


class Tensor:
    def __init__(self, data, requires_grad=False, is_leaf=True, grad_fn=None):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, torch.Tensor):
            if data.is_leaf:
                self.data = data
            else:
                self.data = data.clone().detach()
            self.data.requires_grad = False
        else:
            self.data = torch.tensor(data, requires_grad=False)
        self.requires_grad = requires_grad
        self.grad = None
        self.is_leaf = is_leaf
        self.grad_fn = grad_fn

    def reshape(self, *shape):
        from .autograd.functions.reshape import Reshape  # Local import
        return Reshape.apply(self, shape)

    def view(self, *shape):
        return self.reshape(*shape)

    def sum(self, dim=None, keepdim=False):
        from .autograd.functions.sum import Sum  # Local import
        return Sum.apply(self, dim, keepdim)

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
                if isinstance(grad_input, Tensor) and input_tensor.requires_grad:
                    input_tensor.backward(grad_input)

    @property
    def shape(self):
        return self.data.shape
    
    def size(self, dim=None):
        return self.data.size(dim)
    
    def dim(self):
        return self.data.dim()

    def to(self, device):
        return Tensor(self.data.to(device), requires_grad=self.requires_grad)

    def clone(self):
        return Tensor(self.data.clone(), requires_grad=self.requires_grad)

    def detach(self):
        return Tensor(self.data.detach(), requires_grad=self.requires_grad)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.mul import Mul  # Local import

        return Mul.apply(self, other)

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.add import Add  # Local import

        return Add.apply(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.add import Add

        return Add.apply(self, -other)

    def __rsub__(self, other):
        return other + (-self)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.matmul import MatMul  # Local import

        return MatMul.apply(self, other)

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.matmul import MatMul  # Local import

        return MatMul.apply(other, self)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __ne__(self, other):
        return self.data != other.data

    def transpose(self, dim0, dim1):
        return Tensor(self.data.transpose(dim0, dim1), requires_grad=self.requires_grad)

    def t(self):
        return self.transpose(0, 1)