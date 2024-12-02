# test_linear.py

import torch
from mytorch.tensor import Tensor
from mytorch.nn.modules.linear import Linear
from mytorch.autograd.functions.matmul import MatMul
from mytorch.autograd.functions.add import Add
from mytorch.autograd.functions.transpose import Transpose

# Define a simple Linear layer
in_features = 3
out_features = 4
batch_size = 2

# Initialize custom Linear layer
custom_linear = Linear(in_features, out_features)

# Initialize PyTorch Linear layer with the same weights and biases for comparison
pytorch_linear = torch.nn.Linear(in_features, out_features, bias=True)
pytorch_linear.weight.data = custom_linear.weight.data.clone()
pytorch_linear.bias.data = custom_linear.bias.data.clone()

# Create input tensor
input_data = torch.randn(batch_size, in_features)
input_tensor = Tensor(input_data, requires_grad=True)

# Forward pass using custom Linear
transposed_weight = Transpose.apply(custom_linear.weight, 0, 1)  # (in_features, out_features)
matmul_output = MatMul.apply(input_tensor, transposed_weight)
output_tensor = Add.apply(matmul_output, custom_linear.bias)

# Compute loss as sum of outputs
loss = output_tensor.sum()

# Backward pass
loss.backward()

# Retrieve gradients
custom_input_grad = input_tensor.grad
custom_weight_grad = custom_linear.weight.grad
custom_bias_grad = custom_linear.bias.grad

# Perform the same operations in PyTorch for comparison
input_pt = input_data.clone().detach().requires_grad_(True)
pytorch_output = pytorch_linear(input_pt)
pytorch_loss = pytorch_output.sum()
pytorch_loss.backward()

# Retrieve PyTorch gradients
pytorch_input_grad = input_pt.grad
pytorch_weight_grad = pytorch_linear.weight.grad
pytorch_bias_grad = pytorch_linear.bias.grad

# Compare outputs
assert torch.allclose(output_tensor.data, pytorch_output, atol=1e-6), "Forward pass outputs do not match!"

# Compare gradients
assert torch.allclose(custom_input_grad, pytorch_input_grad, atol=1e-6), "Input gradients do not match!"
assert torch.allclose(custom_weight_grad, pytorch_weight_grad, atol=1e-6), "Weight gradients do not match!"
assert torch.allclose(custom_bias_grad, pytorch_bias_grad, atol=1e-6), "Bias gradients do not match!"

print("Linear Module Test Passed Successfully!")
