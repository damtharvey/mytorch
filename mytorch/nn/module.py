# mytorch/nn/module.py


from mytorch.tensor import Tensor
from collections import OrderedDict


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    @property
    def device(self):
        """
        Returns the device on which the module's parameters and buffers are located.
        If parameters and buffers are spread across multiple devices, raise an exception
        or return None.
        """
        devices = set()

        # Check parameters
        for param in self._parameters.values():
            if param is not None:
                devices.add(param.data.device)

        # Check buffers
        for buffer in self._buffers.values():
            if buffer is not None:
                devices.add(buffer.device)

        if len(devices) == 1:
            # All parameters and buffers are on the same device
            return next(iter(devices))
        elif len(devices) == 0:
            # No parameters or buffers
            return None
        else:
            # Parameters and buffers are spread across multiple devices
            raise RuntimeError("Module parameters and buffers are on different devices.")

    def to(self, device):
        # Update parameters
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = param.to(device)

        # Update buffers
        for name, buffer in self._buffers.items():
            if buffer is not None:
                self._buffers[name] = buffer.to(device)

        # Update submodules
        for name, module in self._modules.items():
            if module is not None:
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
        if name in ("_parameters", "_modules", "_buffers", "training"):
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
