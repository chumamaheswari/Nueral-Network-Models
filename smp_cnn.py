import torch.nn as nn
import torch.optim as optim

# Define a CNN architecture
class SceneCNN(nn.Module):
    def __init__(self, num_classes):
        super(SceneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 64 * 64, num_classes)  # 128x128 → 64x64 after pooling

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x

# Initialize model
num_classes = 10  # CIFAR-10 has 10 classes
model = SceneCNN(num_classes)
print(model)