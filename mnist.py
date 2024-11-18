import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Device Configuration
# Checking if a GPU is available, if not fall back to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess the MNIST dataset
def get_data_loaders(batch_size: int = 64):
    """
    Loads and preprocesses the MNIST dataset. Creates data loaders for training and testing.
    - Normalizes the images to have mean=0.5 and std=0.5.
    - Returns DataLoader objects for training and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL images to PyTorch tensors.
        transforms.Normalize((0.5,), (0.5,))  # Normalize the image pixel values.
    ])
    
    # Download and load the training dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # Download and load the testing dataset
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Define the neural network
class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network with:
    - One hidden layer with 128 neurons and ReLU activation.
    - An output layer with 10 neurons (one for each digit).
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer to hidden layer
        self.relu = nn.ReLU()              # Activation function
        self.fc2 = nn.Linear(128, 10)      # Hidden layer to output layer
    
    def forward(self, x):
        """
        Defines the forward pass:
        - Flattens the input image from (28x28) to (784).
        - Applies a fully connected layer, ReLU activation, and another fully connected layer.
        """
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.relu(self.fc1(x))  # Apply first layer + ReLU
        x = self.fc2(x)  # Apply second layer
        return x

# Train the model
def train_model(model: nn.Module, train_loader, criterion, optimizer, epochs: int = 5):
    """
    Trains the model using the given DataLoader, loss function, and optimizer.
    - Iterates through the training data for the specified number of epochs.
    - Computes the loss, backpropagates, and updates the model's weights.
    """
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Move images and labels to the configured device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradients from the previous step
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights
            running_loss += loss.item()  # Accumulate loss
        
        # Print average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model
def evaluate_model(model: nn.Module, test_loader):
    """
    Evaluates the model on the test data and computes its accuracy.
    - Loops through the test data without computing gradients.
    - Compares predicted labels with true labels to calculate accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)  # Get model predictions
            _, predicted = torch.max(outputs.data, 1)  # Get class with highest score
            total += labels.size(0)  # Total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy

# Test predictions on a small sample
def test_predictions(model: nn.Module, test_loader):
    """
    Tests the model on a batch of test images and displays predictions alongside true labels.
    - Visualizes the first 5 images in the batch with matplotlib.
    """
    model.eval()  # Set the model to evaluation mode
    sample_images, sample_labels = next(iter(test_loader))  # Get a batch of test images
    sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
    
    with torch.no_grad():
        outputs = model(sample_images)  # Get predictions
        _, predictions = torch.max(outputs, 1)  # Get class with highest score
    
    # Display first 5 images with predictions
    for i in range(5):  # Loop over the first 5 images
        plt.imshow(sample_images[i].cpu().squeeze(), cmap="gray")  # Display image
        plt.title(f"True Label: {sample_labels[i].item()}, Predicted: {predictions[i].item()}")
        plt.axis("off")  # Hide axis
        plt.show()

# Main block to tie everything together
def main():
    # Load data
    train_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Initialize the model, loss function, and optimizer
    model = NeuralNetwork().to(device)  # Move model to the device
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer
    
    # Train the model
    print("Training the model...")
    train_model(model, train_loader, criterion, optimizer, epochs=5)
    
    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved as mnist_model.pth")
    
    # Load the model (example of how to reload)
    model.load_state_dict(torch.load("mnist_model.pth"))
    model.eval()
    print("Model loaded and ready for inference.")
    
    # Test predictions
    print("Testing predictions on a small sample...")
    test_predictions(model, test_loader)

if __name__ == "__main__":
    main()
