# mytorch/optim/sgd.py


class SGD:
    def __init__(self, parameters, lr=0.01):
        """
        Initializes the SGD optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
        """
        self.parameters = list(parameters)  # Ensure it's a list for multiple iterations
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        """
        Resets the gradients of all optimized parameters.
        """
        for param in self.parameters:
            param.grad = None
