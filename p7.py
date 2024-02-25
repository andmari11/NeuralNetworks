import math

"""
cuantificar como de equivocado esta el modelo:

L[i]= -sum(target_value[i,j] * log(predicted_values[i,j]))


one-hot=(0: [1, 0, 0],      L[i]=-log(predicted_values[i,k])= (a*1 + b*0 + c*0)
         1: [0, 1, 0],      L[i]=-log(predicted_values[i,k])= (a*0 + b*1 + c*0)
         2: [0, 0, 1])      L[i]=-log(predicted_values[i,k])= (a*0 + b*0 + c*1)

"""

#predicted (wrong???)
softmax_output=[0.7, 0.1, 0.2]
#target (the right answer)
target_output=[1, 0, 0]


loss=-(math.log(softmax_output[0])*target_output[0]+
       math.log(softmax_output[1])*target_output[1]+
       math.log(softmax_output[2])*target_output[2])

#es equivalente a -math.log(softmax_output[0])

print(loss)