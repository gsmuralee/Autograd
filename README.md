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

# Negative log likelihood loss for binary classification used in softmax
def nll_loss(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Cross entropy loss for binary classification used in sigmoid
def cross_entropy_loss(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Mean squared error loss for regression used in linear regression
def mean_squared_error_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Mean absolute error loss for regression used in linear regression
def mean_absolute_error_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
