# Artificial Neural Network from Scratch
### -by Aditi Vyas

### Overview

This project demonstrates how to build an Artificial Neural Network (ANN) from scratch using only NumPy and Pandas. The ANN is trained on the MNIST dataset, a collection of handwritten digits, to classify images into ten categories (0-9). Matplotlib is used to visualize training progress and results.

### Dataset

The model is trained on the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits. You can download the dataset from the following link:

MNIST Dataset - [Digit Recogniser](https://www.kaggle.com/c/digit-recognizer/data)

### Prerequisites

Ensure you have Python installed along with the required libraries:
numpy pandas matplotlib (in requirements.txt)

### Implementation Details

The neural network consists of the following components:

- Input Layer: Accepts 10 flattened 28x28 pixel images (784 features)

- Hidden Layers: Uses 1 fully connected layers with activation functions 

- Output Layer: 10 neurons (one for each digit, using softmax activation)

### Forward Propagation

Multiply inputs by weights and add biases.

Apply activation function (ReLU for hidden layers, Softmax for output).

### Backpropagation

Compute the error using cross-entropy loss.

Compute gradients using the chain rule.

Update weights and biases using gradient descent.

### Training the Model

Normalize the images.

Train the network for a specified number of epochs.

Evaluating the Model

Visualizing Results

### Future Enhancements

- Implement different optimizers (Adam, RMSprop)

- Add dropout for regularization

- Experiment with deeper architectures

- Implement CNNs for better performance