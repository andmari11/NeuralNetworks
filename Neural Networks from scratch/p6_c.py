import math
import numpy as np

#softmax= input -> exponencial-> normalizar -> output

#cual es "más" correcta? todos los valores negativos son 0?-> se necesita otra activation function
layer_outputs=[[4.8, 1.21, 2.385],
               [8.9, -1.81, 0.2],
               [1.41, 1.051, 0.026]]


#exponencial (no perdemos valores negativos (e^x))
exp_values=np.exp(layer_outputs)

"""
normalizar (x/total) 
(axis: 0 => suma en vertical, 1 => suma en horizontal)
keepdims => solucion en dimensiones originales (vertical) 
(True: [[1],        False: [1, 2, 3])
        [2],
        [3]] 

np hace luego broadcast al dividir
"""
#overflow!!!!!!!!!!!
norm_values=exp_values/np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)

#exponencial causa overflow -> e^x - max(layer_outputs)
norm_values=exp_values-np.max(layer_outputs, axis=1, keepdims=True)/np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)





