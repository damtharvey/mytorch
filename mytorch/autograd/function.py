from typing import Tuple
from mytorch.tensor import Tensor
from mytorch.autograd.context import Context


class Function:
    """
    Base class for defining custom autograd operations.

    A `Function` encapsulates the forward and backward computations for an operation.
    Subclasses should implement the `forward` and `backward` methods.

    Attributes:
        ctx (Context): The context object for saving information needed during the backward pass.
        saved_tensors (Tuple[Tensor, ...]): Tensors saved during the forward pass for use in backward computation.
    """

    @classmethod
    def apply(cls, *args: Tensor) -> Tensor:
        """
        Applies the forward computation of the function and sets up the backward computation.

        Args:
            *args: Input tensors for the forward computation.

        Returns:
            Tensor: The output tensor from the forward computation.
        """
        ctx = Context()
        output = cls.forward(ctx, *args)
        func = cls(ctx)
        func.inputs = args  # Save inputs for backward pass
        output.grad_fn = func  # Attach backward computation
        return output

    def __init__(self, ctx: Context, *saved_tensors: Tensor):
        """
        Initializes the function with a context and saved tensors.

        Args:
            ctx: The context object for this operation.
            saved_tensors: Tensors to save for use in the backward pass.
        """
        self.ctx: Context = ctx
        self.saved_tensors: Tuple[Tensor, ...] = saved_tensors

    @staticmethod
    def forward(ctx: Context, *args: Tensor) -> Tensor:
        """
        Computes the forward pass for this operation.

        Args:
            ctx: The context object for saving information needed for the backward pass.
            *args: Input tensors for the forward computation.

        Returns:
            Tensor: The result of the forward computation.
        """
        raise NotImplementedError("Forward method must be implemented in a subclass.")

    def backward(self, grad_output: Tensor) -> Tuple[Tensor, ...]:
        """
        Computes the backward pass for this operation.

        Args:
            grad_output: The gradient of the loss with respect to the output of this operation.

        Returns:
            Tuple[Tensor, ...]: Gradients of the loss with respect to the inputs.
        """
        raise NotImplementedError("Backward method must be implemented in a subclass.")
