from typing import Iterable
from mytorch.tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Attributes:
        parameters (List[Tensor]): The list of parameters to optimize.
        lr (float): The learning rate for the optimizer.
    """

    def __init__(self, parameters: Iterable[Tensor], lr: float = 0.01):
        """
        Initializes the SGD optimizer.

        Args:
            parameters: An iterable of `Tensor` objects to optimize.
            lr: The learning rate for parameter updates. Default is 0.01.
        """
        self.parameters: list[Tensor] = list(parameters)  # Ensure parameters is a list for iteration
        self.lr: float = lr

    def step(self) -> None:
        """
        Performs a single optimization step, updating each parameter by subtracting the
        gradient scaled by the learning rate.
        """
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self) -> None:
        """
        Sets the gradients of all parameters to None, clearing any previous gradients.
        """
        for param in self.parameters:
            param.grad = None
