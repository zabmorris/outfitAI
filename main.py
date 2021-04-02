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
        self.weights1   = np.random.rand(self.input.shape[1],6) 
        self.weights2   = np.random.rand(6,2)                 
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
        self.weights2 = self.weights2 - 0.005*d_weights2
        self.weights1 = self.weights1 - 0.005*d_weights1


##########################################################################
###                     Starting point of the program                  ###
##########################################################################

# provide the input data and output data for training
x = np.array([[-9.35,69.5,8.5,0,0,4,4.5,-22],[11.14,91.5,28,29.6,0,1,6.5,11.14],[24.05,74,16.5,0,0,10,4.5,39],[18.64,81.5,11.5,0,0,7,4.5,27],[-7.7,78,19.5,0,7.4,3,4.5,-20],[22.75,63.5,23.5,0,0,9,2.5,32],[-0.09,79.5,12,0,0,1,8,-4],[18.54,63,7.5,0,0,9,2.5,18.54],[25,75.5,16,0,0,9,3.5,40],[4.2,88.5,25,1.4,0,1,4,-5],[-1.25,80.5,18,0,0,3,4.5,-9],[0.7,87,22.5,0,0.8,1,4.5,-8],[6.35,59.5,19.5,0,0,7,4.5,6.35],[11.55,76,22,17.2,0,2,7.5,11.55],[22.2,90,20.5,25.6,0,3,6.5,35],[22.2,77,18,0,0,6,4.5,33],[8.15,81,14,4.4,0,3,6.5,8.15],[8.94,85.5,11.5,0.6,0,4,4.5,8.94],[0.89,81,17,0,0,2,4,-7],[-1.6,59,16.5,0,0,4,3.5,-9],[17.45,64,14.5,0,0,9,5,26],[7.19,79,8.5,0.4,0,3,7.5,7.19],[-0.54,69,23.5,0,0,1,5.5,-11],[12.05,71.5,9.5,0,0,6,2,12.05],[17.5,90.5,24,7,0,6,6,30],[9.94,99,23,15.2,0,2,6.5,9.94],[4.5,82,21.5,12,0,2,4.5,-4],[10.89,65.5,15,0,0,4,4,10.89],[19.14,71.5,17.5,0.6,0,9,4.5,32],[-5.15,39.5,19.5,0,0,5,4,-15],[5.6,80.5,31,5.2,0,5,7,5.6],[20.85,69,9.5,0,0,8,4.5,29],[5.4,82.5,15,0,0,2,5.5,5.4],[7.05,65.5,12.5,0,0,5,4,7.05],[16.64,75.5,17,1.2,0,6,5,30],[1.8,70.5,14.5,0,0,1,4.5,-6],[-0.19,59.5,20.5,0,0,3,4,-8],[-0.29,94.5,12.5,2.6,0,1,6.5,-6],[24.1,85,7,0,0,8,4,41],[17.85,68,16,0,0,8,3,26],[7.25,76.5,12,3,0,7,4.5,7.25],[-20.29,64,28,0,0,2,4,-37],[10.3,74,11.5,15.6,0,6,4,10.3],[24.3,73.5,10,0,0,10,3.5,24.3],[12.5,90,11,0.2,0,7,5.5,12.5],[-10.25,84.5,9.5,0,0,2,4.5,-19],[18.6,72.5,8.5,0,0,6,4,30],[-5.5,59,24.5,0,0,3,1,-18],[18.35,75,7,0,0,9,4,30],[20.6,68.5,17.5,0,0,9,4.5,33],[-4.65,73,30.5,0,0,1,7.5,-15],[1.04,74.5,23,0.4,0,1,4.5,-11],[15.9,69.5,14.5,0,0,7,1.5,15.9],[20.39,77,19.5,1.6,0,7,4,31],[-1.14,74.5,18,0,0,2,5,-12],[26.45,78,14,13,0,8,4,26.45],[-3.34,69,16.5,0,0,5,3.5,-13],[8.69,82,25.5,0,0,5,3.5,8.69],[10.8,68,33,0,0,7,5,10.8],[23.65,60,17.5,0,0,9,4,23.65],[12.5,90,11,0.2,0,7,5.5,-2],[24.3,73.5,10,0,0,10,3.5,33],[10.3,74,11.5,15.6,0,6,4,10.3],[-20.29,64,28,0,0,2,4,-20.29],[7.25,76.5,12,3,0,7,4.5,-1],[17.85,68,16,0,0,8,3,17.85],[24.1,85,7,0,0,8,4,24.1],[-0.29,94.5,12.5,2.6,0,1,6.5,-15],[-0.19,59.5,20.5,0,0,3,4,-0.19],[1.8,70.5,14.5,0,0,1,4.5,1.8],[16.64,75.5,17,1.2,0,6,5,25],[7.05,65.5,12.5,0,0,5,4,-8],[5.4,82.5,15,0,0,2,5.5,5.4],[20.85,69,9.5,0,0,8,4.5,20.85],[5.6,80.5,31,5.2,0,5,7,-7]])

y_desired = np.array([[1,0.5],[0.5,0.5],[0,0],[0,0],[1,1],[0,0],[1,0.5],[0,0.5],[0,0],[1,0.5],[1,0.5],[1,0.5],[1,0.5],[0.5,0.5],[0,0],[0,0],[0.5,0.5],[0.5,0.5],[1,1],[1,1],[0,0],[0.5,0.5],[1,1],[0.5,0.5],[0,0.5],[0.5,0.5],[1,0.5],[0.5,0.5],[0,0.5],[1,1],[1,0.5],[0,0],[1,0.5],[0.5,0.5],[1,0.5],[1,0.5],[1,0.5],[1,0.5],[0,0],[0,0],[0.5,0.5],[1,0.5],[0.5,0.5],[0,0],[0.5,0],[1,0.5],[0,0],[1,0.5],[0,0],[0,0],[1,0.5],[0.5,0.5],[0.5,0.5],[0,0],[1,1],[0,0],[1,1],[0.5,0.5],[0.5,0.5],[0,0],[1,0.5],[0,0],[0.5,0.5],[1,1],[1,0.5],[0,0],[0,0],[1,1],[0,0.5],[0.5,0.5],[0,0],[1,0.5],[0.5,0.5],[0,0],[1,0.5]])
ANN = NeuralNetwork(x, y_desired)

# create an empty array to hold values of loss function in each iteration
loss_values = []

# train the ANN for 3000 iterations 
for i in range(100000):
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