import torch
from mytorch.nn.module import Module
from mytorch.tensor import Tensor
from mytorch.nn.modules.linear import Linear

# Custom module with parameters, buffers, and submodules
class MyModule(Module):
    def __init__(self):
        super().__init__()
        self._parameters['param1'] = Tensor(torch.randn(2, 2))
        self._parameters['param2'] = Tensor(torch.randn(3))
        self._buffers['buffer1'] = Tensor(torch.zeros(3))
        self._modules['submodule'] = Linear(2, 2)

# Initialize module
module = MyModule()

# Initial device (all on CPU)
print("Initial Device (param1):", module._parameters['param1'].data.device)
print("Initial Device (submodule):", module._modules['submodule'].weight.data.device)

# Move module to CUDA
module.to('cuda')

# Verify device transfer
print("After .to('cuda') (param1):", module._parameters['param1'].data.device)  # Expected: cuda:0
print("After .to('cuda') (submodule):", module._modules['submodule'].weight.data.device)  # Expected: cuda:0

# Move module back to CPU
module.to('cpu')

# Verify device transfer
print("After .to('cpu') (param1):", module._parameters['param1'].data.device)  # Expected: cpu
print("After .to('cpu') (submodule):", module._modules['submodule'].weight.data.device)  # Expected: cpu
