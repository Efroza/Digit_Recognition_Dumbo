import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import cv2
import numpy as np
from PIL import Image
import os

# Define transformations to apply to the images
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the MNIST training and validation datasets
train_dataset = datasets.MNIST('data/', train=True, transform=transform, download=True)
val_dataset = datasets.MNIST('data/', train=False, transform=transform, download=True)

# Create data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Initialize the model, optimizer, and loss function
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training the model
def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs=5):
    if "mnist_cnn_model.pth" in os.listdir():
        print("Model already trained!")
        return
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Accuracy: {(100 * correct / total):.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), 'mnist_cnn_model.pth')

# Loading and recognizing digits
def load_and_recognize_digit(image_path):
    # Load the trained CNN model
    model.load_state_dict(torch.load('mnist_cnn_model.pth'))
    model.eval()

    # Preprocess function for OpenCV image
    def preprocess_opencv_image(img):
        img = img[0:480, 0:800]
        cv2.imwrite("drawing.png", img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        img_normalized = img_resized / 255.0
        return img_normalized

    # Function to perform digit recognition
    def recognize_digit(image):
        transformed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = transformed(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.item()

    # OpenCV for image input
    image = cv2.imread(image_path)  
    preprocessed_image = preprocess_opencv_image(image)
    predicted_digit = recognize_digit(preprocessed_image)

    return predicted_digit

def launch():
    train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs=5)

def recognize_digits():
    return load_and_recognize_digit("drawing.png")