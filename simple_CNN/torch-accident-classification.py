# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from CNN import CNN
from time import perf_counter
from tqdm import tqdm
from torch.utils.data import Subset
import random

def get_subset(dataset, fraction=0.1):
    size = int(len(dataset) * fraction)
    indices = random.sample(range(len(dataset)), size)
    return Subset(dataset, indices)


batch_size = 32
img_height = 224
img_width = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('../sim_dataset/data/train', transform=transform)
val_dataset = datasets.ImageFolder('../sim_dataset/data/val', transform=transform)
test_dataset = datasets.ImageFolder('../sim_dataset/data/test', transform=transform)

class_names = train_dataset.classes

train_dataset = get_subset(train_dataset, fraction=0.3)
val_dataset = get_subset(val_dataset, fraction=0.1)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




# %%
model = CNN(len(class_names)).to(device)

# %%
# Loss + optimizer
criterion = nn.CrossEntropyLoss()  # includes softmax internally
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# %%
# Training loop
epochs = 10
best_val_acc = 0

train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model_weights.pth")
        print("Saved best model")

# %%
# Plot training curves
plt.plot(train_loss_hist, label='training loss')
plt.plot(val_loss_hist, label='validation loss')
plt.legend()
plt.grid(True)
plt.savefig("TrainValLosses")
plt.show()

plt.plot(train_acc_hist, label='training accuracy')
plt.plot(val_acc_hist, label='validation accuracy')
plt.legend()
plt.grid(True)
plt.savefig("TrainValAccuracy")
plt.show()

# %%
# Load best model
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# %%
# Visualization on test data
plt.figure(figsize=(30, 30))

images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

outputs = model(images)
_, preds = torch.max(outputs, 1)

for i in range(40):
    ax = plt.subplot(10, 4, i + 1)
    
    img = images[i].cpu().permute(1, 2, 0).numpy()
    plt.imshow(img)
    
    plt.title(f'Pred: {class_names[preds[i]]} | Actl: {class_names[labels[i]]}')
    plt.axis('off')
    plt.grid(True)

plt.savefig("Predictions")
plt.show()
