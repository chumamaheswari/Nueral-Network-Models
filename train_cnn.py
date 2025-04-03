import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Example dataset (replace with actual dataset)
x_train = torch.randn(100, 10)  # 100 samples, 10 features
y_train = torch.randint(0, 2, (100,))  # 100 labels (binary classification)
x_val = torch.randn(20, 10)
y_val = torch.randint(0, 2, (20,))

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Ensure model is properly initialized before use
class YourModelClass(nn.Module):
    def __init__(self):
        super(YourModelClass, self).__init__()
        self.layer = nn.Linear(10, 2)  # Example layer, adjust as needed

    def forward(self, x):
        return self.layer(x)

model = YourModelClass()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total if total > 0 else 0  # Avoid division by zero
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}")

# Train the model
train_model(model, train_loader, val_loader, optimizer, criterion)
