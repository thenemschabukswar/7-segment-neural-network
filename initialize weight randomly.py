'''
	Warning: This code resets the weights to zero! Use with caution
	Weight	W1: 45 x  7
	Weight	W2: 7  x 10
	Bias	B1: 1  x  7
	Bias	B2: 1  x 10
'''

import numpy as np

W1 = np.random.random_sample((45,7))
W2 = np.random.random_sample((7,10))
B1 = np.random.random_sample((7))
B2 = np.random.random_sample((1,10))
np.savetxt("W1.csv", W1, delimiter=",")
np.savetxt("W2.csv", W2, delimiter=",")
np.savetxt("B1.csv", B1, delimiter=",")
np.savetxt("B2.csv", B2, delimiter=",")

