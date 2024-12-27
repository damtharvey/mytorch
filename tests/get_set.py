from mytorch.tensor import Tensor

tensor = Tensor([1, 2, 3], requires_grad=True)

x = tensor[:2]

x.sum().backward()

print(tensor.grad)  # Expected: [1, 1, 0]

