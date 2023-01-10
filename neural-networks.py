import torch
import torch.nn as nn  # Network, loss functions
import torch.optim as optim  # Optimizers (Adam, SGD, etc.)
import torch.nn.functional as F  # Activation functions (ReLU, etc.)
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.stack = nn.Sequential(
          nn.Linear(input_size, 512),
          nn.ReLU(),
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.stack(x)


# model = NN(28 * 28, 10)
# num_examples = 64 # mini-batch size
# x = torch.randn(num_examples, 28 * 28)
# print(model(x).shape)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28 * 28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
  print("Epoch {}/{}".format(epoch+1, num_epochs))
  for batch_idx, (data, targets) in enumerate(train_loader):
    # Get data to cuda if possible
    data = data.to(device=device)
    targets = targets.to(device=device)

    # Get to correct shape (batch_size, 1, 28, 28) -> (batch_size, 1*28*28)
    data = data.reshape(data.shape[0], -1)

    # Forward
    scores = model(data)
    loss = criterion(scores, targets)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Gradient descent or adam step
    optimizer.step()

# Check accuracy
def check_accuracy(loader, model):
  if loader.dataset.train:
    print('Checking accuracy on training data')
  else:
    print('Checking accuracy on test data')
  num_correct = 0
  num_samples = 0
  model.eval()

  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device)
      y = y.to(device=device)
      x = x.reshape(x.shape[0], -1)

      scores = model(x)
      _, predictions = scores.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)

    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%')
  
  model.train()

# Run
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)