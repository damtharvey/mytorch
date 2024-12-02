import torch
from mytorch.tensor import Tensor
from mytorch.nn.modules.loss import CrossEntropyLoss

# Create dummy data
batch_size = 3
num_classes = 5
input_data = torch.randn(batch_size, num_classes, requires_grad=True)
target_data = torch.tensor([1, 0, 4], dtype=torch.int64)

# Convert to custom Tensor
input = Tensor(input_data.detach().clone(), requires_grad=True)
target = target_data  # Targets can remain as torch tensors since they don't require gradients

# Instantiate loss function
criterion = CrossEntropyLoss()

# Forward pass
loss = criterion(input, target)

# Backward pass
loss.backward()

# Print the loss and gradients
print("Custom MyTorch Loss:", loss.data.item())
print("Custom MyTorch Input Gradients:")
print(input.grad)

# PyTorch equivalent for comparison
input_data = input_data.clone().detach().requires_grad_(True)
criterion_torch = torch.nn.CrossEntropyLoss()
loss_torch = criterion_torch(input_data, target_data)
loss_torch.backward()

print("\nPyTorch Loss:", loss_torch.item())
print("PyTorch Input Gradients:")
print(input_data.grad)
