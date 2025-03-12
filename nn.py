import random
from engine import Value

class Module:
#reset gradients 
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
#Neuron class creates a neuron with random weights, bias, and activation function which is tanh
class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        # summation of weights and inputs + bias
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act
    
    def parameters(self):
        return self.w + [self.b]

#Layer class creates a layer of neurons with a given number of inputs
class Layer(Module):
    #initilizes the layer with a given number of inputs for Neuron and number of neurons
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self, x):
        #applies the neuron to the input
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

#Sequential class creates a sequential model with a given list of layers
class MLP(Module):
    #initilizes the MLP with a given number of inputs and outputs
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        #creates a list of layers with the given number of inputs and outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        #applies the layers to the input and passes the output to the next layer
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    