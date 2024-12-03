# mytorch/autograd/function.py

from mytorch.autograd.context import Context


class Function:
    @classmethod
    def apply(cls, *args):
        ctx = Context()
        output = cls.forward(ctx, *args)
        func = cls(ctx)
        func.inputs = args
        output.grad_fn = func
        return output

    def __init__(self, ctx, *saved_tensors):
        self.ctx = ctx
        self.saved_tensors = saved_tensors

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError
