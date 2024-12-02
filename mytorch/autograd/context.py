# mytorch/autograd/context.py


class Context:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
