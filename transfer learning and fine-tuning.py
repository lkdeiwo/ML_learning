# imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 28
sequence_length = 28
num_layers = 1
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 2

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# load pretrain model & modify it
model = torchvision.models.vgg16(pretrained=True)
model.avgpool = Identity()
model.classifier = nn.Linear(512, 10)
model.to(device)

# load data
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

# check accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device).squeeze(1)
                y = y.to(device=device)

                scores = model(x)
                _, prediction = scores.max(1)
                num_correct += (prediction == y).sum()
                num_samples += prediction.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)