from typing import Tuple
from mytorch.tensor import Tensor


class Context:
    """
    A utility class for storing information needed during the backward pass of autograd operations.

    Attributes:
        saved_tensors (Tuple[Tensor, ...]): Tensors saved for use during the backward pass.
    """

    def __init__(self):
        """
        Initializes a new Context object with an empty tuple of saved tensors.
        """
        self.saved_tensors: Tuple[Tensor, ...] = ()

    def save_for_backward(self, *tensors: Tensor) -> None:
        """
        Saves tensors for use in the backward pass.

        Args:
            tensors: Tensors to save for backward computation.
        """
        self.saved_tensors = tensors
