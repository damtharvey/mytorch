# examples/train_mnist.py

from mytorch.tensor import Tensor
from mytorch.nn import Module
from mytorch.nn.modules import Linear, ReLU, Reshape
from mytorch.optim import SGD
from mytorch.nn.modules.loss import CrossEntropyLoss

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleNN(Module):
    def __init__(self):
        super().__init__()
        self.reshape = Reshape(-1)
        # self.fc1 = Linear(28 * 28, 128)
        self.fc1 = Linear(784, 10)
        # self.relu1 = ReLU()
        # self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.fc2(x)
        return x


def main():
    num_epochs = 1
    batch_size = 1

    # Load MNIST data
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model = SimpleNN()
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            inputs = Tensor(images, requires_grad=True)
            targets = labels  # You may need to convert labels to a compatible Tensor

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
