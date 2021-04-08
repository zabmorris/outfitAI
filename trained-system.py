# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:09:44 2021

@author: zachary
"""

import numpy as np
#import matplotlib.pyplot as plt

# define sigmoid function as activation function for both hidden layer 
# and output layer
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

# define the first-order derivative of a sigmoid function
def sigmoid_derivative(sigma):
    return sigma * (1.0 - sigma)

# define sum-of-squares error function as loss function
def compute_certain(y_actual, y_desired):
    return (1-(0.5*(y_desired - y_actual)**2))*100

# define and initialize an ANN with one input lary, one hidden layer, 
# and one output layer
class NeuralNetwork:
    def __init__(self, x):
        self.input      = x
        self.weights1   = [[-3.69294924e-01, 8.55715944e-02, -1.13989213e+00, -4.72262632e-01, 1.25315839e+00,  3.79629295e-01,  6.83439260e-01,  3.75211143e+00, 3.79665742e-01,  1.67875386e+00,  1.25318047e+00,  1.65117334e+01, -4.84973789e-01,  1.31350892e+00,  3.73602715e+00,  1.31321826e+00, 8.77593706e+00,  3.79671845e-01],[ 9.84062550e+00,  3.64103851e-01,  3.00378833e+01,  1.16981263e+00, 2.02955123e+00,  1.06811866e+00, 6.37473462e-01, -9.61084149e+00, 1.06813525e+00, -1.02654163e+02,  2.02945103e+00, -3.02985093e+01, 3.37379448e+00,  1.09724063e+00, -1.19150561e+00,  1.09712279e+00, 5.13602624e+00,  1.06813804e+00]] 
        self.weights2   = [[-13.64789519],[ -8.5646116 ],[-60.88123059],[ 22.75524476],[-15.91581467],[ 11.84314575],[ 39.71582204],[ 66.25695352],[ 11.84883407],[-72.38085781],[-16.38271311],[-26.19238837],[ 54.4107515 ],[-21.31239515],[ 26.59108104],[-14.66623543],[  4.09802179],[ 11.84978642]]                 
        #self.y          = y_desired
        self.output     = np.zeros(1)

# define a function to calculate the actual output of ANN 
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

# define BP algorithm to update the weights of hidden layer and the weights
# of output layer

    #def backprop(self):
        # application of the chain rule to find derivative of the 
        # loss function with respect to weights2 and weights1
       # d_weights2 = np.dot(self.layer1.T, 
             #(self.output - self.y) * sigmoid_derivative(self.output))
        
       # temp = np.dot((self.output - self.y)*sigmoid_derivative(self.output),
      #                                                      self.weights2.T)
                    
       # d_weights1 = np.dot(self.input.T,temp*sigmoid_derivative(self.layer1))

        # update the weights with the derivative (slope) of the loss function
       # self.weights2 = self.weights2 - 0.00151*d_weights2
       # self.weights1 = self.weights1 - 0.00151*d_weights1


##########################################################################
###                     Starting point of the program                  ###
##########################################################################

# provide the input data and output data for training
#x = np.array([[-22,0],[11.14,29.6],[39,0],[27,0],[-20,0],[32,0],[-4,0],[18.54,0],[40,0],[-5,1.4],[-9,0],[-8,0],[6.35,0],[11.55,17.2],[35,25.6],[33,0],[8.15,4.4],[8.94,0.6],[-7,0],[-9,0],[26,0],[7.19,0.4],[-11,0],[12.05,0],[30,7],[9.94,15.2],[-4,12],[10.89,0],[32,0.6],[-15,0],[5.6,5.2],[29,0],[5.4,0],[7.05,0],[30,1.2],[-6,0],[-8,0],[-6,2.6],[41,0],[26,0],[7.25,3],[-37,0],[10.3,15.6],[24.3,0],[12.5,0.2],[-19,0],[30,0],[-18,0],[30,0],[33,0],[-15,0],[-11,0.4],[15.9,0],[31,1.6],[-12,0],[26.45,13],[-13,0],[8.69,0],[10.8,0],[23.65,0],[-2,0.2],[33,0],[10.3,15.6],[-20.29,0],[-1,3],[17.85,0],[24.1,0],[-15,2.6],[-0.19,0],[1.8,0],[25,1.2],[-8,0],[5.4,0],[20.85,0],[-7,5.2],[-3,0],[-9,0],[13.25,0],[-7,0],[-25,0],[30,0.4],[12,10.2],[-4,0],[-5,0],[7.4,0],[-8,0],[-14,0],[33,0],[11.15,0],[-6,7],[13.7,0],[7.2,0],[-16,0],[-6,0.2],[-14,0],[-7,0],[27,0],[-10,0],[-7,0],[4.95,12.6],[38,0.6],[-14,0],[35,40.8],[-15,0],[-21,0],[16.45,18.5],[-5,37.6],[-21,0],[-6,0.2],[-10,0],[28,3],[35,1.6],[-13,0],[17.89,0],[-11,0],[-7,0],[-5,11.4],[10.39,0],[8.75,4.2],[14.75,0.2],[11.89,12],[-15,0],[-6,0],[-6,6.2],[6.5,22.2],[-16,0],[-10,0],[7.5,5.6],[41,0],[14.4,0],[37,18],[6.6,0],[-9,0],[31,0]])
#y_desired = np.array([[0.66],[1],[0],[0],[0.66],[0],[0.66],[0],[0],[1],[0.66],[0.66],[0.33],[1],[1],[0],[1],[1],[0.66],[0.66],[0],[1],[0.66],[0.33],[1],[1],[1],[0.33],[1],[0.66],[1],[0],[0.33],[0.33],[1],[0.66],[0.66],[1],[0],[0],[1],[0.66],[1],[0],[1],[0.66],[0],[0.66],[0],[0],[0.66],[0.66],[0],[0],[0.66],[1],[0.66],[0.33],[0.33],[0],[0.66],[0],[1],[0.66],[0.66],[0],[0],[0.66],[0.66],[0.66],[0],[0.66],[0.33],[0],[1],[0.66],[0.66],[0.33],[0.66],[0.66],[0],[1],[0.66],[0.66],[0.33],[0.66],[0.66],[0],[0.33],[1],[0.33],[0.33],[0.66],[0.66],[0.66],[0.66],[0],[0.66],[0.66],[1],[1],[0.66],[1],[0.66],[0.66],[1],[1],[0.66],[1],[0.66],[0],[0],[0.66],[0],[0.66],[0.66],[1],[0.33],[0.33],[0.33],[1],[0.66],[0.66],[1],[1],[0.66],[0.66],[1],[0],[0.33],[1],[0.33],[0.66],[0]])

feelsLike = float(input("Feels like temp: "))
rain = float (input("Predicted downfall (mm): "))
x = [feelsLike, rain] 
ANN = NeuralNetwork(x)

# create an empty array to hold values of loss function in each iteration
#loss_values = []

# train the ANN for 3000 iterations 
#for i in range(250000000):
ANN.feedforward()
y = ANN.output
    #ANN.backprop()
    #loss = compute_loss(ANN.output, y_desired)
    #loss_values.append(loss)
    
###########################################################################
certainty = 0

if (y<=0.165):
    certainty = compute_certain(y, 0)
    print("outfitAI reccomends summer clothing today with {}% certainty!".format(certainty))
elif (y<=0.495 and y>0.165):
    certainty = compute_certain(y, 0.33)
    print("outfitAI reccomends spring clothing today with {}% certainty!".format(certainty))
elif (y<=0.825 and y>0.495):
    certainty = compute_certain(y, 0.66)
    print("outfitAI reccomends winter clothing today with {}% certainty!".format(certainty))
else:
    certainty = compute_certain(y, 1)
    print("outfitAI reccomends dressing for rain today with {}% certainty!".format(certainty))

# print out the actual output of final ineration 
#for i,y in enumerate(ANN.output):
#print('y= {}'.format(y))

   
#print('\nInput Weights:\n {}'.format(ANN.weights1))
#print('\nOutput Weights:\n {}'.format(ANN.weights2))
 
 
# print out the value of loss function in final iteration
# print out the graph of loss function of all iterations 
#plt.plot(loss_values)
#plt.show()