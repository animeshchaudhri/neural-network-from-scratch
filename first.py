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