import numpy as np
import matplotlib.pyplot as plt


#crea datos aleatorios (puntos, nespirales)
def createData(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

#data y target 
X, y= createData(100,3)
#plt.scatter(X[:,0],X[:,1])
#plt.scatter(X[:,0],X[:,1],c=y, cmap="brg")
plt.show()


#filas
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 *np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases

#funcion de activacion
class ActivationReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


#inputs, neuronas
layer1 = LayerDense(2,5)
#X*weight+bias
layer1.forward(X)

#actuvamos funcion
activation1=ActivationReLu()
activation1.forward(layer1.output)

print(activation1.output)

