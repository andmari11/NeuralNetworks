import numpy as np;

#3 neuron with 4 inputs
inputs = [1, 2, 3, 2.5]


weights=[[0.2, 0.8, -0.5, 1],
         [0.5, -0.91, 0.26, -0.5],
         [-0.26, -0.27, 0.17, 0.87]]

biases= [2, 3, 0.5]


layer_output=[]

"""
"""

#calculo de cada neurona
for neuron_weights, neuron_bias in zip(weights, biases):

    #output de cada peso y su input + bias
    neuron_output=np.dot(inputs, weights) + neuron_bias

    layer_output.append(neuron_output)

print(layer_output)