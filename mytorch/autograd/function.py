# mytorch/autograd/function.py

from mytorch.autograd.context import Context


class Function:
    @classmethod
    def apply(cls, *args):
        ctx = Context()
        output = cls.forward(ctx, *args)
        # Create an instance to store ctx and inputs
        func = cls()
        func.ctx = ctx
        func.inputs = args
        output.grad_fn = func
        return output

    def __init__(self, ctx, *saved_tensors):
        self.ctx = ctx
        self.saved_tensors = saved_tensors

    def backward(self, grad_output):
        # To be implemented in subclasses
        raise NotImplementedError
