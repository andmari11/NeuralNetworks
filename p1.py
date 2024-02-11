inputs = [1.2, 5.1, 2.1]
weights= [3.1, 2.1, 8.7]
bias = 3
output = bias

for i in range(3):   
    output += inputs[i]*weights[i]

print(output)