#3 neuron with 4 inputs
inputs = [1, 2, 3, 2.5]

weights1= [0.2, 0.8, -0.5, 1]
weights2= [0.5, -0.91, 0.26, -0.5]
weights3= [-0.26, -0.27, 0.17, 0.87]


bias1 = 2
bias2 =3
bias3 =0.5
output={bias1, bias2, bias3}

for i in range(4):
    output[0] += inputs[i]*weights1[i]
    output[1] += inputs[i]*weights1[i]
    output[2] += inputs[i]*weights1[i]

print(output)