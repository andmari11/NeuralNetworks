import math

#cual es "más" correcta? -> se necesita otra activation function
layer_outputs1=[4.8, 1.21, 2.385]
layer_outputs2=[4.8, 4.79, 4.25]

exp_values=[]

#no perdemos valores negativos
for output in layer_outputs1:
    exp_values.append(math.e**output)

#total
norm_base=sum(exp_values)
norm_values=[]

for value in exp_values:
    norm_values.append(value/norm_base)
    print(str(value) + "/" + str(norm_base) + "= "+ str(value/norm_base))

print("total= " + str(sum(norm_values)))



