import torch
from mytorch import Tensor
from mytorch.nn.modules import Reshape

input_tensor = Tensor(torch.randn(2, 3, 4), requires_grad=True)
reshape_module = Reshape(-1)
output_tensor = reshape_module(input_tensor)
print(output_tensor.data.shape)  # Output: (24,)

# Verify backward pass
loss = output_tensor.sum()
loss.backward()
print(input_tensor.grad.shape)  # Output: (2, 3, 4)
