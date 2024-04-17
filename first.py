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

#how to reduce dimensions
#pca, t-sne, autoencoders
#what is pca
# principal component analysis
#what is t-sne
# t-distributed stochastic neighbour embedding
#what is autoencoders
# autoencoders are a type of neural network that learns to compress data and then reconstruct it back to its original form.
#in pca what is the objective
# maximise variance
#what is variance
# variance is the measure of the spread of the data points

#in t-sne what is the objective
# minimise distance between similar points and maximise distance between dissimilar points
#in autoencoders what is the objective
# reduce dimensions and then increase dimensions

#what does pca does to ooirginal features
# it reduces the dimensions of the original features
#what does t-sne does to original features
# it reduces the dimensions of the original features
#what does autoencoders does to original features

#what is the main advatnage of using dimensionality reduction techniques
# it reduces the dimensions of the data and makes it easier to visualize and interpret the data.
#List of dimensionality reduction technique are singular value decomposition
# Ica
# Pca
# projection
# mainfold

# which dimensionality reduction technique is used for visualizing the data in a lower dimensional space
# t-sne is used for visualizing the data in a lower dimensional space
# what is the main advantage of using t-sne
# it is used to visualize the data in a lower dimensional space
# what is the main disadvantage of using t-sne
# it is computationally expensive
# what is the main advantage of using pca
# it is used to reduce the dimensions of the data
# what is the main disadvantage of using pca
# in pca what is variance ration
# variance ration is the ratio of the variance of the principal components to the total variance of the data
# in pca what is explained variance

