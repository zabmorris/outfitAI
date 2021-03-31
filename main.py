import numpy as np
import matplotlib.pyplot as plt
import requests

# define sigmoid function as activation function for both hidden layer 
# and output layer
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

# define the first-order derivative of a sigmoid function
def sigmoid_derivative(sigma):
    return sigma * (1.0 - sigma)

# define sum-of-squares error function as loss function
def compute_loss(y_actual, y_desired):
    return ((y_desired - y_actual)**2).sum()

# define and initialize an ANN with one input lary, one hidden layer, 
# and one output layer
class NeuralNetwork:
    def __init__(self, x, y_desired):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],8) 
        self.weights2   = np.random.rand(8,8)                 
        self.y          = y_desired
        self.output     = np.zeros(self.y.shape)

# define a function to calculate the actual output of ANN 
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

# define BP algorithm to update the weights of hidden layer and the weights
# of output layer

    def backprop(self):
        # application of the chain rule to find derivative of the 
        # loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, 
             2*(self.output - self.y) * sigmoid_derivative(self.output))
        
        temp = np.dot(2*(self.output - self.y)*sigmoid_derivative(self.output),
                                                            self.weights2.T)
                    
        d_weights1 = np.dot(self.input.T,temp*sigmoid_derivative(self.layer1))

        # update the weights with the derivative (slope) of the loss function
        self.weights2 = self.weights2 - d_weights2
        self.weights1 = self.weights1 - d_weights1


##########################################################################
###                     Starting point of the program                  ###
##########################################################################

# provide the input data and output data for training
x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y_desired = np.array([[0],[1],[1],[0]])
ANN = NeuralNetwork(x, y_desired)

# create an empty array to hold values of loss function in each iteration
loss_values = []

# train the ANN for 3000 iterations 
for i in range(3000):
    ANN.feedforward()
    ANN.backprop()
    loss = compute_loss(ANN.output, y_desired)
    loss_values.append(loss)
    
###########################################################################

# print out the actual output of final ineration 
for i,y in enumerate(ANN.output):
    print('y{} = {}'.format(i+1, y[0]))

   
print('\nInput Weights:\n {}'.format(ANN.weights1))
print('\nOutput Weights:\n {}'.format(ANN.weights2))
 
 
# print out the value of loss function in final iteration
print(f"\nFinal loss: {loss}")
# print out the graph of loss function of all iterations 
plt.plot(loss_values)
plt.show()