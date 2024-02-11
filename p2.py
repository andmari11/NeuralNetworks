inputs = [1, 2, 3]
weights= [0.2, 0.8, -0.5]
bias = 2
output = 0

for i in range(3):   
    output += inputs[i]*weights[i]

output+=bias
print(output)