import torch
from mytorch.tensor import Tensor
from mytorch.autograd.functional.loss import CrossEntropyLoss

# Test Case 1: Single Data Point (1D Input)
input_1d = Tensor(torch.tensor([2.0, 1.0, 0.1]), requires_grad=True)
target_1d = torch.tensor(0, dtype=torch.int64)

# Forward pass
loss_1d = CrossEntropyLoss.apply(input_1d, target_1d)
print("1D Loss:", loss_1d.data)

# Backward pass
loss_1d.backward()
print("1D Input Gradient:", input_1d.grad)

# Test Case 2: Batch of Data Points (2D Input)
input_2d = Tensor(torch.tensor([[2.0, 1.0, 0.1], [0.5, 0.2, 0.3]]), requires_grad=True)
target_2d = torch.tensor([0, 2], dtype=torch.int64)

# Forward pass
loss_2d = CrossEntropyLoss.apply(input_2d, target_2d)
print("2D Loss:", loss_2d.data)

# Backward pass
loss_2d.backward()
print("2D Input Gradient:", input_2d.grad)

# PyTorch comparison
import torch.nn.functional as F

# PyTorch CrossEntropyLoss for comparison
input_1d_pt = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
target_1d_pt = torch.tensor(0, dtype=torch.int64)
loss_1d_pt = F.cross_entropy(input_1d_pt.unsqueeze(0), target_1d_pt.unsqueeze(0))  # 1D input
loss_1d_pt.backward()
print("1D PyTorch Loss:", loss_1d_pt.item())
print("1D PyTorch Gradient:", input_1d_pt.grad)

input_2d_pt = torch.tensor([[2.0, 1.0, 0.1], [0.5, 0.2, 0.3]], requires_grad=True)
target_2d_pt = torch.tensor([0, 2], dtype=torch.int64)
loss_2d_pt = F.cross_entropy(input_2d_pt, target_2d_pt)  # 2D input
loss_2d_pt.backward()
print("2D PyTorch Loss:", loss_2d_pt.item())
print("2D PyTorch Gradient:", input_2d_pt.grad)
