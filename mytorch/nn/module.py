# mytorch/nn/module.py


from mytorch.tensor import Tensor
from collections import OrderedDict


class Module:
    def __init__(self):
        object.__setattr__(self, '_parameters', OrderedDict())
        object.__setattr__(self, '_modules', OrderedDict())
        object.__setattr__(self, '_buffers', OrderedDict())
        object.__setattr__(self, 'training', True)

    def to(self, device):
        for param in self.parameters():
            param.to(device)
        for module in self._modules.values():
            module.to(device)
        return self

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self):
        for param in self.parameters():
            param.grad = None

    def __setattr__(self, name, value):
        if name in ('_parameters', '_modules', '_buffers', 'training'):
            object.__setattr__(self, name, value)
        elif isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
            if name in self._modules:
                del self._modules[name]
            object.__setattr__(self, name, value)
        elif isinstance(value, Module):
            self._modules[name] = value
            if name in self._parameters:
                del self._parameters[name]
            object.__setattr__(self, name, value)
        else:
            if name in self._parameters:
                del self._parameters[name]
            if name in self._modules:
                del self._modules[name]
            object.__setattr__(self, name, value)



    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented.")
