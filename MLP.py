import random
from autograd import Value

class Neuron:
    def __init__(self, N):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(N)]
        self.bias = Value(random.uniform(-1, 1))
        
        
    def __call__(self, x : Value):
        # w * x + b : w * x is a dot product
        Z = sum((w * x for w, x in zip(self.weights, x)) , self.bias)
        A = Z.tanh()
        return A
        

class Layer:
    def __init__(self, n_input : int, n_output: int):
        self.neurons = [Neuron(n_input) for _ in range(n_output) ]
    
    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
        
        
    
    def __repr__(self):
        return f"Number of neurons: {len(self.neurons)}"