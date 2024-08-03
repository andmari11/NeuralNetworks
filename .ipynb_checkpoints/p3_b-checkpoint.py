import numpy as np;

#3 neuron with 4 inputs
inputs = [1, 2, 3, 2.5]


weights=[[0.2, 0.8, -0.5, 1],
         [0.5, -0.91, 0.26, -0.5],
         [-0.26, -0.27, 0.17, 0.87]]

biases= [2, 3, 0.5]


layer_output=[]

"""
ERROR= np.dot(inputs, weights)
ValueError: shapes (4,) and (3,4) not aligned: 4 (dim 0) != 3 (dim 0)

A = m x n
B = p x q

n tiene que ser igual a p
"""

layer_output= np.dot(weights, inputs)+biases

print(layer_output)