# mytorch/nn/module.py

from mytorch.tensor import Tensor


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self):
        for param in self.parameters():
            param.grad = None

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented.")
