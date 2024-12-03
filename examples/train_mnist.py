# examples/train_mnist.py

import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mytorch.tensor import Tensor
from mytorch.nn import Module
from mytorch.nn.modules import Linear, ReLU, Reshape
from mytorch.optim import SGD
from mytorch.nn.modules.loss import CrossEntropyLoss


class SimpleNN(Module):
    def __init__(self):
        super().__init__()
        self.reshape = Reshape()
        self.fc1 = Linear(28 * 28, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.reshape(x, (-1, 28 * 28))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 1
    batch_size = 64
    learning_rate = 0.1

    # Load MNIST data
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model = SimpleNN().to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    n_examples = 0
    n_correct = 0
    train_loss = 0

    for epoch in range(num_epochs):
        for images, labels in (progress_bar := tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs = Tensor(images, requires_grad=True).to(device)
            targets = labels.to(device)  # You may need to convert labels to a compatible Tensor

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            n_examples += labels.size(0)
            n_correct += (outputs.data.argmax(1) == targets).sum().item()
            train_loss += loss.data

            progress_bar.set_postfix_str(f"Loss: {train_loss / n_examples:.4f}, Accuracy: {n_correct / n_examples:.4f}")


if __name__ == "__main__":
    main()
