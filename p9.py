import matplotlib.pyplot as plt
import numpy as np
import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()

#100 puntos y 3 clases de puntos
#X,y= vertical_data(samples=100,classes=3)
X,y= spiral_data(samples=100,classes=3)

plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap='brg')
plt.show()

#filas de neuronas con pesos y bias randoms
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 *np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases

#activaction relu function
class ActivationReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#softmax
class ActivationSoftmax:
    def forward(self, inputs):
        exp_values= np.exp(inputs-np.max(inputs,axis=1, keepdims=True))
        norm_values= exp_values/np.sum(exp_values,axis=1, keepdims=True)
        self.output=norm_values

#cuantificar el error
class Loss:
    def calculate(self, output, y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss

class LossCategoricalCrossentropy(Loss):
    #y_pred: valores de neural network, y_true: valores target
    def forward(self,y_pred, y_true):
        samples=len(y_pred)
        #para evitar log(0)->inf
        y_pred_clipped=np.clip(y_pred, 1e-7, 1-1e-7)

        #comprobamos si one-hot(matriz binaria) o escalar (guarda el indice del targetcorrecto)
        if len(y_true.shape)==1:
            correct_confidences=y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape)==2:
            correct_confidences=np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods=-np.log(correct_confidences)
        return negative_log_likelihoods

dense1=LayerDense(2, 3)
activation1=ActivationReLu()
dense2=LayerDense(3, 3)
activation2=ActivationSoftmax()

loss_function=LossCategoricalCrossentropy()

#ret(se ponen valores predetermindaos)
lowest_loss=99999999
best_dense1_weights=dense1.weights.copy()
best_dense1_biases=dense1.biases.copy()
best_dense2_weights=dense2.weights.copy()
best_dense2_biases=dense2.biases.copy()


for i in range(100000):

    dense1.weights+=0.05 * np.random.randn(2, 3)
    dense1.biases+=0.05 * np.random.randn(1, 3)
    dense2.weights+=0.05 * np.random.randn(3, 3)
    dense2.biases+=0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss=loss_function.calculate(activation2.output, y)

    predictions=np.argmax(activation2.output, axis=1)
    accuracy=np.mean(predictions==y)

    if loss < lowest_loss:
        print("New set of weights in i:", i, "loss:", loss, "acc:", accuracy)
        lowest_loss=loss
        best_dense1_weights=dense1.weights.copy()
        best_dense1_biases=dense1.biases.copy()
        best_dense2_weights=dense2.weights.copy()
        best_dense2_biases=dense2.biases.copy()
    else:
        dense1.weights=best_dense1_weights.copy()
        dense1.biases=best_dense1_biases.copy()
        dense2.weights=best_dense2_weights.copy()
        dense2.biases=best_dense2_biases.copy()

    
