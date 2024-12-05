from __future__ import annotations

from typing import Any, Optional, Union, Tuple
import torch


class Tensor:
    """
    A wrapper around `torch.Tensor` with support for custom automatic differentiation.

    Attributes:
        data (torch.Tensor): The underlying data of the tensor.
        requires_grad (bool): Whether this tensor requires gradient computation.
        grad (Optional[torch.Tensor]): The gradient of the tensor.
        is_leaf (bool): Whether this tensor is a leaf tensor (does not have a grad_fn).
        grad_fn (Optional[Any]): The function that generated this tensor.
    """

    def __init__(
        self,
        data: Union[Tensor, torch.Tensor, Any],
        requires_grad: bool = False,
        is_leaf: bool = True,
        grad_fn: Optional[Any] = None,
    ):
        """
        Initializes the Tensor object.

        Args:
            data: The data for the tensor, which can be another Tensor, a torch.Tensor, or a compatible input.
            requires_grad: Whether the tensor requires gradients.
            is_leaf: Whether the tensor is a leaf tensor.
            grad_fn: The function that generated this tensor.
        """
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
        self.requires_grad: bool = requires_grad
        self.grad: Optional[torch.Tensor] = None
        self.is_leaf: bool = is_leaf
        self.grad_fn: Optional[Any] = grad_fn

    def reshape(self, *shape: int) -> Tensor:
        """
        Reshapes the tensor to the specified shape.

        Args:
            shape: The new shape for the tensor.

        Returns:
            A reshaped tensor.
        """
        from .autograd.functions.reshape import Reshape  # Local import

        return Reshape.apply(self, shape)

    def view(self, *shape: int) -> Tensor:
        """
        An alias for `reshape`.

        Args:
            shape: The new shape for the tensor.

        Returns:
            A reshaped tensor.
        """
        return self.reshape(*shape)

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        """
        Computes the sum of the tensor elements along the specified dimension.

        Args:
            dim: The dimension along which to sum.
            keepdim: Whether to retain the reduced dimensions.

        Returns:
            A tensor containing the sum.
        """
        from .autograd.functions.sum import Sum  # Local import

        return Sum.apply(self, dim, keepdim)

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """
        Computes the gradient of the tensor with respect to its inputs.

        Args:
            grad_output: The gradient of the output tensor. Defaults to a tensor of ones.
        """
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
    def shape(self) -> torch.Size:
        """Returns the shape of the tensor."""
        return self.data.shape

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        """Returns the size of the tensor."""
        return self.data.size(dim)

    def dim(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return self.data.dim()

    @property
    def device(self) -> torch.device:
        """Returns the device on which the tensor is stored."""
        return self.data.device

    def to(self, device: Union[torch.device, str]) -> Tensor:
        """
        Moves the tensor to the specified device.

        Args:
            device: The target device.

        Returns:
            A new tensor on the target device.
        """
        return Tensor(self.data.to(device), requires_grad=self.requires_grad)

    def clone(self) -> Tensor:
        """Returns a clone of the tensor."""
        return Tensor(self.data.clone(), requires_grad=self.requires_grad)

    def detach(self) -> Tensor:
        """Returns a detached version of the tensor."""
        return Tensor(self.data.detach(), requires_grad=self.requires_grad)

    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        """Element-wise multiplication of two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.mul import Mul  # Local import

        return Mul.apply(self, other)

    def __neg__(self) -> Tensor:
        """Negates the tensor."""
        return self * -1

    def __add__(self, other: Union[Tensor, float, int]) -> Tensor:
        """Element-wise addition of two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.add import Add  # Local import

        return Add.apply(self, other)

    def __radd__(self, other: Union[Tensor, float, int]) -> Tensor:
        """Right-side addition."""
        return self + other

    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor:
        """Element-wise subtraction."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.add import Add

        return Add.apply(self, -other)

    def __rsub__(self, other: Union[Tensor, float, int]) -> Tensor:
        """Right-side subtraction."""
        return other + (-self)

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication of two tensors."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.matmul import MatMul  # Local import

        return MatMul.apply(self, other)

    def __rmatmul__(self, other: Tensor) -> Tensor:
        """Right-side matrix multiplication."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        from .autograd.functions.matmul import MatMul  # Local import

        return MatMul.apply(other, self)

    def __repr__(self) -> str:
        """Returns the string representation of the tensor."""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self) -> str:
        """Returns the string representation of the tensor."""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __getitem__(self, key: Any) -> torch.Tensor:
        """Returns the element(s) at the specified index."""
        return self.data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Sets the element(s) at the specified index."""
        self.data[key] = value

    def __len__(self) -> int:
        """Returns the length of the tensor."""
        return len(self.data)

    def __iter__(self) -> iter:
        """Returns an iterator over the tensor elements."""
        return iter(self.data)

    def __eq__(self, other: Tensor) -> torch.Tensor:
        """Checks equality between two tensors."""
        return self.data == other.data

    def __ne__(self, other: Tensor) -> torch.Tensor:
        """Checks inequality between two tensors."""
        return self.data != other.data

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        """Transposes the specified dimensions of the tensor."""
        return Tensor(self.data.transpose(dim0, dim1), requires_grad=self.requires_grad)

    def t(self) -> Tensor:
        """Returns the transpose of a 2D tensor."""
        return self.transpose(0, 1)
