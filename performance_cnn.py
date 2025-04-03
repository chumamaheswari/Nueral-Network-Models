import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define a simple CNN model (Replace with your trained model)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 2)  # Assuming 2 classes

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x

# Initialize model
model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0  # Avoid division by zero

# Example dataset
dummy_data = torch.randn(10, 3, 32, 32)  # 10 samples, 3 channels, 32x32 images
dummy_labels = torch.randint(0, 2, (10,))  # Binary classification (0 or 1)
test_dataset = TensorDataset(dummy_data, dummy_labels)
test_loader = DataLoader(test_dataset, batch_size=2)

# Evaluate model
test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot results
plt.bar(["Test Accuracy"], [test_accuracy])
plt.ylabel("Accuracy")
plt.title("CNN Model Performance")
plt.show()
