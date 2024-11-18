# MNIST-Classification-Project 
Using the MNIST dataset, this Python project creates a basic feedforward neural network for handwritten digit classification. The network offers an introduction to deep learning using PyTorch by being trained to identify and predict numbers (0–9) based on their pixel representation.
# features 
- Device Compatibility: If a GPU is available, it is automatically detected and used; if not, the CPU is used.
- An example of a feedforward neural network is one that has:

    ReLU activation and 128 neurons make up one hidden layer.
    Ten neurons (one for each digit) make up the output layer.

- Instruction and Assessment:

    use an optimizer and loss function to train on the MNIST dataset.
    assesses the accuracy of the model using a test dataset.

Visualization: Shows genuine labels next to predictions on a small sample of test photos.

# Getting Started 
Requirements:- 
Make sure you have Python installed along with the following libraries:
    torch
    torchvision
    matplotlib 
  You can install these libraries using pip: pip install torch torchvision matplotlib
# Usage Running the Code
The script performs the following tasks:
- Downloads the MNIST dataset and preprocesses it.
- Trains the neural network for 5 epochs (default).
- Evaluates the model's accuracy on the test set.
- Saves the trained model to disk for later use.
- Visualizes predictions on a small batch of test samples.

# OutPut 
Epoch 1/5, Loss: 0.3001
Epoch 2/5, Loss: 0.1205 
with 98,20% accuracy the displays test images is true and predicted the correct labels.


