'''
Program Flow:
	-> Open CSV file containing weights of neural network
	-> Convert it to matrix format
	-> Load images and convert to matrix format
	-> Train network using appropriate algorithm
	-> Update weights[directly to csv file] after every 20 epochs
	-> Exit when loss is very close to zero
'''

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from numpy import genfromtxt

W1 = genfromtxt('W1.csv', delimiter=',')
W2 = genfromtxt('W2.csv', delimiter=',')
B1 = genfromtxt('B1.csv', delimiter=',')
B2 = genfromtxt('B2.csv', delimiter=',')

#initialise images as zero

zero = np.zeros((1,45))
one = np.zeros((1,45))
two = np.zeros((1,45))
three = np.zeros((1,45))
four = np.zeros((1,45))
five = np.zeros((1,45))
six = np.zeros((1,45))
seven = np.zeros((1,45))
eight = np.zeros((1,45))
nine = np.zeros((1,45))

def input_image():

	zero  = cv2.imread('0.png',0).reshape((1, 45))

	one   = cv2.imread('1.png',0).reshape((1, 45))

	two	  = cv2.imread('2.png',0).reshape((1, 45))

	three = cv2.imread('3.png',0).reshape((1, 45))

	four  = cv2.imread('4.png',0).reshape((1, 45))

	five  = cv2.imread('5.png',0).reshape((1, 45))

	six   = cv2.imread('6.png',0).reshape((1, 45))

	seven = cv2.imread('7.png',0).reshape((1, 45))

	eight = cv2.imread('8.png',0).reshape((1, 45))

	nine  = cv2.imread('9.png',0).reshape((1, 45))

input_image()

X_train = np.zeros((10,45))
X_train[0] = zero
X_train[1] = one
X_train[2] = two
X_train[3] = three
X_train[4] = four
X_train[5] = five
X_train[6] = six
X_train[7] = seven
X_train[8] = eight
X_train[9] = nine
Y_train = np.array([[1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,1]])

loss = {}
#FF class
class FeedForward:
  
  def __init__(self, W1, W2, B1, B2):
    self.W1 = W1.copy()			#W1 -> (45,7)
    self.W2 = W2.copy()			#W2 -> (7,10)
    self.B1 = B1.copy()	
    self.B2 = B2.copy()
  
  def sigmoid(self, X):
    return 1.0/(1.0 + np.exp(-X))
  
  def softmax(self, X):
    exps = np.exp(X)
    return exps / np.sum(exps, axis=1).reshape(-1,1)
  
  def forward_pass(self, X):
    self.A1 = np.matmul(X,self.W1) + self.B1 # (10, 45) * (45, 7) -> (10, 7)
    self.H1 = self.sigmoid(self.A1) # (10, 7)
    self.A2 = np.matmul(self.H1, self.W2)  # (10, 7) * (7, 10) -> (10, 10)
    self.H2 = self.softmax(self.A2) # (10, 10)
    return self.H2
    
  def grad_sigmoid(self, X):
    return X*(1-X) 
  
  def grad(self, X, Y):
    self.forward_pass(X)
    m = X.shape[0]
    
    self.dA2 = self.H2 - Y # (10, 10) - (10, 10) -> (10, 10)
    
    self.dW2 = np.matmul(self.H1.T, self.dA2) # (7, 10) * (10, 10) -> (7, 10)
    self.dB2 = np.sum(self.dA2, axis=0).reshape(1, -1) # (10, 10) -> (1, 10)
    self.dH1 = np.matmul(self.dA2, self.W2.T) # (10, 10) * (10, 7) -> (10, 7)
    self.dA1 = np.multiply(self.dH1, self.grad_sigmoid(self.H1)) # (10, 7) .* (10, 7) -> (10, 7)
    
    self.dW1 = np.matmul(X.T, self.dA1) # (45,10) * (10, 7) -> (45, 7)
    self.dB1 = np.sum(self.dA1, axis=0).reshape(1, -1) # (10, 7) -> (1, 7)

      
  def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=True):
    #if display_loss:
    #  #loss = {}
    for i in range(epochs):
      self.grad(X, Y) # X -> (10, 45), Y -> (10, 10)
        
      m = X.shape[0]
      self.W2 = self.W2 - learning_rate * (self.dW2/m)
      self.B2 = self.W2 - learning_rate * (self.dB2/m)
      self.W1 = self.W1 - learning_rate * (self.dW1/m)
      self.B1 = self.B1 - learning_rate * (self.dB1/m)

      if display_loss:
        Y_pred = self.predict(X)
        loss[i] = log_loss(np.argmax(Y, axis=1), Y_pred)
    
    
    #if display_loss:
    #  plt.plot(loss.values())
    #  plt.xlabel('Epochs')
    #  plt.ylabel('Log Loss')
    #  plt.show()
    
  
  def predict(self, X):
    Y_pred = self.forward_pass(X)
    return np.array(Y_pred).squeeze()
p1 = FeedForward(W1,W2,B1,B2)
ron = 20
while(ron>0):
	print ("\nStarting epochs")
	p1.fit(X_train,Y_train,epochs=20,learning_rate=.5)
	print ("\nSaving Weights and Biases")
	np.savetxt("W1.csv", W1, delimiter=",")
	np.savetxt("W2.csv", W2, delimiter=",")
	np.savetxt("B1.csv", B1, delimiter=",")
	np.savetxt("B2.csv", B2, delimiter=",")
	ron = ron-1
	

#X_train = np.array([[zero]], [[one]], [[two]], [[three]], [[four]], [[five]], [[six]], [[seven]], [[eight]], [[nine]])
'''
X_train = np.zeros((10,45))
X_train[0] = zero
X_train[1] = one
X_train[2] = two
X_train[3] = three
X_train[4] = four
X_train[5] = five
X_train[6] = six
X_train[7] = seven
X_train[8] = eight
X_train[9] = nine
Y_train = np.array([[1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,1]])
X_pred = np.zeros

print (X_train.shape, Y_train.shape)
'''