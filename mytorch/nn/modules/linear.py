import torch

from ...tensor import Tensor
from ..module import Module
from ...autograd.functions.matmul import MatMul
from ...autograd.functions.add import Add
from ...autograd.functions.transpose import Transpose
from ..init import kaiming_uniform


class Linear(Module):
    """
    A fully connected linear layer.

    This layer applies a linear transformation to the input: `y = x @ weight.T + bias`.

    Attributes:
        weight (Tensor): The learnable weights of shape `(out_features, in_features)`.
        bias (Tensor): The learnable bias of shape `(out_features,)`.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initializes the Linear layer with the specified input and output dimensions.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
        """
        super().__init__()
        self._parameters["weight"] = Tensor(torch.empty(out_features, in_features), requires_grad=True)
        self._parameters["bias"] = Tensor(torch.zeros(out_features), requires_grad=True)

        # Initialize weights and biases
        kaiming_uniform(self._parameters["weight"])
        bound = 1 / in_features**0.5 if in_features > 0 else 0
        self._parameters["bias"].data.uniform_(-bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """
        Computes the forward pass of the Linear layer.

        Args:
            input (Tensor): The input tensor of shape `(N, in_features)`, where `N` is the batch size.

        Returns:
            Tensor: The output tensor of shape `(N, out_features)`.
        """
        weight = self._parameters["weight"]
        transposed_weight = Transpose.apply(weight, 0, 1)  # (in_features, out_features)
        bias = self._parameters["bias"]
        out = MatMul.apply(input, transposed_weight)
        out = Add.apply(out, bias)
        return out

    @property
    def weight(self) -> Tensor:
        """
        Returns:
            Tensor: The learnable weights of the Linear layer.
        """
        return self._parameters["weight"]

    @property
    def bias(self) -> Tensor:
        """
        Returns:
            Tensor: The learnable biases of the Linear layer.
        """
        return self._parameters["bias"]
