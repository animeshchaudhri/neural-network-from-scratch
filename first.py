import numpy as np

x = [[3.1,2.1,8.7,0.1],
    [4.1,2.5,9.8,98],
    [-0.26,0.67,0.4,0.2]],

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(x)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

# step function is if your output is greater than 0.5, then it is 1, else 0
# sigmoid function is 1/(1+e^-x) more granular than step function
# tanh function is (e^x - e^-x)/(e^x + e^-x)
# relu function is max(0, x)
# softmax function is e^x/sum(e^x)


# what is activation function ?
# activation function is a function that is used to introduce non-linearity in the output of a neuron.
# It is also called transfer function.
# It is used to decide whether a neuron should be activated or not.
# It is used to map the input between the required values.
# It is used to introduce non-linearity in the output of a neuron.
# It is used to make the network learn complex patterns in the data.
# the activation function would be differnt for different layers.

# what is purpose of activation function ?
# The purpose of activation function is to introduce non-linearity in the output of a neuron.
# mimic firing of neurons in the brain.

# what is loss in neural network ?
# loss is the error in the output of the neural network. It is the difference between the predicted output and the actual output.
