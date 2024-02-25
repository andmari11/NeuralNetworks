import numpy as np

#cual es "más" correcta? todos los valores negativos son 0?-> se necesita otra activation function
layer_outputs1=[4.8, 1.21, 2.385]
layer_outputs2=[4.8, 4.79, 4.25]

#exponencial (no perdemos valores negativos (e^x))
exp_values=np.exp(layer_outputs1)

#normalizar (x/total)
norm_values=exp_values/np.sum(exp_values)

#output
print(exp_values)
print(norm_values)
print("total= " + str(np.sum(norm_values)))

#softmax= input -> exponencial-> normalizar -> output


