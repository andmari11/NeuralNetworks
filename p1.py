#1 single neuron with 4 inputs
inputs = [1.2, 5.1, 2.1]
weights= [3.1, 2.1, 8.7]
bias = 3
output = 0

for i in range(3):   
    output += inputs[i]*weights[i]

output+=bias
print(output)