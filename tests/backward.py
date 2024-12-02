# train.py

import torch
from mytorch.tensor import Tensor
from mytorch.optim.sgd import SGD
from mytorch.nn.module import Module
from mytorch.nn.modules import  Linear, ReLU
from mytorch.nn.modules.loss import CrossEntropyLoss

class SimpleNN(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(28*28, 10)

    def forward(self, x, targets):
        x = self.fc1(x)
        return CrossEntropyLoss()(x, targets)
    
class TorchSimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 10)

    def forward(self, x, targets):
        x = self.fc1(x)
        return torch.nn.CrossEntropyLoss()(x, targets)

# Dummy data
inputs = torch.randn(2, 28 * 28)  # Batch size 2
targets = torch.tensor([1, 0], dtype=torch.int64)

# Convert to MyTorch Tensor
inputs = Tensor(inputs, requires_grad=False)
targets = targets  # Targets can remain as torch tensors

# Initialize network and optimizer
model = SimpleNN()
optimizer = SGD(model.parameters(), lr=0.01)

# Forward pass
loss = model(inputs, targets)

# Backward pass
loss.backward()

# Torch equivalent
torch_model = TorchSimpleNN()
torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)

# Copy parameters
torch_model.fc1.weight.data = model.fc1.weight.data
torch_model.fc1.bias.data = model.fc1.bias.data

# Forward pass
torch_inputs = inputs.data.clone().detach().requires_grad_(True)
torch_targets = targets.clone().detach().requires_grad_(False)
torch_loss = torch_model(torch_inputs, torch_targets)

# Backward pass
torch_loss.backward()

# Compare the gradients
print("Max difference in fc1 weight gradients:", torch.max(torch.abs(model.fc1.weight.grad - torch_model.fc1.weight.grad)))
print("Max difference in fc1 bias gradients:", torch.max(torch.abs(model.fc1.bias.grad - torch_model.fc1.bias.grad)))

# Compare the loss
print("Max difference in loss:", torch.abs(loss.data - torch_loss.data))


