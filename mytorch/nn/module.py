from __future__ import annotations
from typing import Any, Iterator, Optional, Union
import torch
from mytorch.tensor import Tensor
from collections import OrderedDict


class Module:
    """
    Base class for all neural network modules.

    A `Module` organizes parameters, submodules, and buffers, and provides methods
    for moving data to devices, clearing gradients, and performing forward passes.

    Attributes:
        _parameters (OrderedDict[str, Tensor]): Parameters of the module.
        _modules (OrderedDict[str, Module]): Submodules of the module.
        _buffers (OrderedDict[str, Tensor]): Buffers (non-parameter tensors) of the module.
        training (bool): Whether the module is in training mode.
    """

    def __init__(self) -> None:
        """
        Initializes the Module with empty parameters, submodules, and buffers.
        """
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    @property
    def device(self) -> Optional[Union[str, torch.device]]:
        """
        Returns the device where the module's parameters and buffers reside.

        Returns:
            Optional[Union[str, torch.device]]: The device if all parameters and buffers
            are on the same device, or None if no parameters or buffers exist.

        Raises:
            RuntimeError: If parameters and buffers are on different devices.
        """
        devices = set()

        # Check devices of parameters
        for param in self._parameters.values():
            if param is not None:
                devices.add(param.data.device)

        # Check devices of buffers
        for buffer in self._buffers.values():
            if buffer is not None:
                devices.add(buffer.device)

        # Handle multiple or no devices
        if len(devices) == 1:
            return next(iter(devices))
        elif len(devices) == 0:
            return None
        else:
            raise RuntimeError("Module parameters and buffers are on different devices.")

    def to(self, device: Union[str, torch.device]) -> Module:
        """
        Moves the module's parameters and buffers to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda').

        Returns:
            Module: The module itself, updated in-place.
        """
        # Move parameters
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = param.to(device)

        # Move buffers
        for name, buffer in self._buffers.items():
            if buffer is not None:
                self._buffers[name] = buffer.to(device)

        # Recursively move submodules
        for name, module in self._modules.items():
            if module is not None:
                module.to(device)

        return self

    def parameters(self) -> Iterator[Tensor]:
        """
        Yields all parameters of the module and its submodules.

        Returns:
            Iterator[Tensor]: An iterator over all parameters.
        """
        # Yield parameters from the current module
        for param in self._parameters.values():
            yield param

        # Recursively yield parameters from submodules
        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self) -> None:
        """
        Sets the gradients of all parameters in the module and its submodules to None.
        """
        for param in self.parameters():
            param.grad = None

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Custom attribute setting for parameters, submodules, and buffers.

        Args:
            name: The name of the attribute.
            value: The value to set. Can be a Tensor, Module, or any other type.
        """
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Calls the module's `forward` method with the given arguments.

        Args:
            *args: Positional arguments for the `forward` method.
            **kwargs: Keyword arguments for the `forward` method.

        Returns:
            Any: The result of the `forward` method.
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        The forward pass of the module. Must be implemented by subclasses.

        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.

        Returns:
            Any: The output of the forward pass.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Forward method not implemented.")
