import numpy as np

softmax_outputs= np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

#0(dog): 1er valor => softmax[i,0]
#1(cat): 2do valor => softmax[i,1]
#2(human): 3er valor => softmax[i,2]
class_targets=[0, 1, 1]

#discernimos de los datos no importantes
useful_values=softmax_outputs[range(len(softmax_outputs)),class_targets]
#calculamos el error
neg_log=-np.log(useful_values)
#calculamos media
avg_loss=np.mean(neg_log)

#calculamos si el mayor valor de softmax coincide con target
predictions=np.argmax(softmax_outputs, axis=1)
accuracy=np.mean(predictions==class_targets)