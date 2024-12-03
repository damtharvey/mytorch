import torch
from mytorch.tensor import Tensor
from mytorch.nn.module import Module


class MyModule(Module):
    def __init__(self):
        super().__init__()
        self._parameters["weight"] = Tensor(torch.randn(3, 3).to("cuda"))
        self._parameters["bias"] = Tensor(torch.zeros(3).to("cuda"))
        self._buffers["running_mean"] = Tensor(torch.zeros(3).to("cuda"))


# Single device (CUDA)
print("\nExpected output: cuda:0")
module = MyModule()
print("Device:", module.device)

# Inconsistent devices
print("\nExpected: RuntimeError")
module._parameters["bias"] = Tensor(torch.zeros(3).to("cpu"))
try:
    print("Device:", module.device)
except RuntimeError as e:
    print("Error:", e)

# No parameters or buffers
print("\nExpected output: None")
module._parameters = {}
module._buffers = {}
print("Device:", module.device)
