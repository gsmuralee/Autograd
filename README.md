Implementing Micrograd from Andrej Karpathy's lecture

Points to note:
    - In the network, each node has data and gradient
    - The Final output is considered as a loss function
    - The loss function is the difference between the predicted output and the actual output
    - gradient is the derivative of the loss function with respect to its data
    - chain rule is used to calculate the gradient of the loss function with respect to its data
    - Gradient is a vector pointed in the direction of the loss function
    - Backpropagation is used to calculate the gradient of the loss function with respect to its data
    - Gradient descent is used to update the weights of the network
    - Gradient is adjuested to minimize the loss function, when the gradient is more the loss function is more and vice versa
