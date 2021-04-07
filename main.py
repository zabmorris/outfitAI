import numpy as np
import matplotlib.pyplot as plt

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
        self.weights1   = np.random.rand(self.input.shape[1],15) 
        self.weights2   = np.random.rand(15,1)                 
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
        self.weights2 = self.weights2 - 0.0009*d_weights2
        self.weights1 = self.weights1 - 0.0009*d_weights1


##########################################################################
###                     Starting point of the program                  ###
##########################################################################

# provide the input data and output data for training
x = np.array([[-22,0],[11.14,29.6],[39,0],[27,0],[-20,0],[32,0],[-4,0],[18.54,0],[40,0],[-5,1.4],[-9,0],[-8,0],[6.35,0],[11.55,17.2],[35,25.6],[33,0],[8.15,4.4],[8.94,0.6],[-7,0],[-9,0],[26,0],[7.19,0.4],[-11,0],[12.05,0],[30,7],[9.94,15.2],[-4,12],[10.89,0],[32,0.6],[-15,0],[5.6,5.2],[29,0],[5.4,0],[7.05,0],[30,1.2],[-6,0],[-8,0],[-6,2.6],[41,0],[26,0],[7.25,3],[-37,0],[10.3,15.6],[24.3,0],[12.5,0.2],[-19,0],[30,0],[-18,0],[30,0],[33,0],[-15,0],[-11,0.4],[15.9,0],[31,1.6],[-12,0],[26.45,13],[-13,0],[8.69,0],[10.8,0],[23.65,0],[-2,0.2],[33,0],[10.3,15.6],[-20.29,0],[-1,3],[17.85,0],[24.1,0],[-15,2.6],[-0.19,0],[1.8,0],[25,1.2],[-8,0],[5.4,0],[20.85,0],[-7,5.2],[-3,0],[-9,0],[13.25,0],[-7,0],[-25,0],[30,0.4],[12,10.2],[-4,0],[-5,0],[7.4,0],[-8,0],[-14,0],[33,0],[11.15,0],[-6,7],[13.7,0],[7.2,0],[-16,0],[-6,0.2],[-14,0],[-7,0],[27,0],[-10,0],[-7,0],[4.95,12.6],[38,0.6],[-14,0],[35,40.8],[-15,0],[-21,0],[16.45,18.5],[-5,37.6],[-21,0],[-6,0.2],[-10,0],[28,3],[35,1.6],[-13,0],[17.89,0],[-11,0],[-7,0],[-5,11.4],[10.39,0],[8.75,4.2],[14.75,0.2],[11.89,12],[-15,0],[-6,0],[-6,6.2],[6.5,22.2],[-16,0],[-10,0],[7.5,5.6],[41,0],[14.4,0],[37,18],[6.6,0],[-9,0],[31,0]])
y_desired = np.array([[0.66],[1],[0],[0],[0.66],[0],[0.66],[0],[0],[1],[0.66],[0.66],[0.33],[1],[1],[0],[1],[1],[0.66],[0.66],[0],[1],[0.66],[0.33],[1],[1],[1],[0.33],[1],[0.66],[1],[0],[0.33],[0.33],[1],[0.66],[0.66],[1],[0],[0],[1],[0.66],[1],[0],[1],[0.66],[0],[0.66],[0],[0],[0.66],[0.66],[0],[0],[0.66],[1],[0.66],[0.33],[0.33],[0],[0.66],[0],[1],[0.66],[0.66],[0],[0],[0.66],[0.66],[0.66],[0],[0.66],[0.33],[0],[1],[0.66],[0.66],[0.33],[0.66],[0.66],[0],[1],[0.66],[0.66],[0.33],[0.66],[0.66],[0],[0.33],[1],[0.33],[0.33],[0.66],[0.66],[0.66],[0.66],[0],[0.66],[0.66],[1],[1],[0.66],[1],[0.66],[0.66],[1],[1],[0.66],[1],[0.66],[0],[0],[0.66],[0],[0.66],[0.66],[1],[0.33],[0.33],[0.33],[1],[0.66],[0.66],[1],[1],[0.66],[0.66],[1],[0],[0.33],[1],[0.33],[0.66],[0]])
ANN = NeuralNetwork(x, y_desired)

# create an empty array to hold values of loss function in each iteration
loss_values = []

# train the ANN for 3000 iterations 
for i in range(10000):
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