import torch
from mytorch.tensor import Tensor

# Create a tensor
input_tensor = Tensor(torch.randn(2, 3), requires_grad=True)

# Perform sum operation
output_tensor = input_tensor.sum()

# Perform backward pass
output_tensor.backward()

# Print gradients
print("Custom MyTorch Input Gradients:")
print(input_tensor.grad)

# For comparison, use PyTorch
input_data = input_tensor.data.clone().detach().requires_grad_(True)
output_data = input_data.sum()
output_data.backward()
print("\nPyTorch Input Gradients:")
print(input_data.grad)
