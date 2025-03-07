import random
from autograd import Value

class Neuron:
    def __init__(self, N):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(N)]
        self.bias = Value(random.uniform(-1, 1))
        
        
    def __call__(self, x : Value):
        # w * x + b : w * x is a dot product
        Z = sum((w * x for w, x in zip(self.weights, x)) , self.bias)
        A = Z.sigmoid()
        return A
    
    def update(self, lr):
        for w in self.weights:
            w.value -= lr * w.gradient
            
    def zero_grad(self):
        for w in self.weights:
            w.zero_grad()
        self.bias.zero_grad()
        

class Layer:
    def __init__(self, n_input : int, n_output: int):
        self.neurons = [Neuron(n_input) for _ in range(n_output) ]
    
    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
    
    def update(self, lr):
        for neuron in self.neurons:
            neuron.update(lr)
            
    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()
        
    def __repr__(self):
        return f"Number of neurons: {len(self.neurons)}"
    
    
class MLP:
    def __init__(self, input_size : int, layer_sizes : list):
        l = [input_size] + layer_sizes
        self.layers = []
        for i in range(len(l) - 1):
            self.layers.append(Layer(l[i], l[i + 1]))
            
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    # I assume that the input data fits into memory so no batch size
    def train(self, x, epochs = None, loss_function = None, lr = 0.001):
        if (loss_function == None):
            def loss(y_pred, y_true):
                inner = [(y1 - y2) ** 2 for y1, y2 in zip(y_pred, y_true)]
                return sum(inner) / len(x)            
            loss_function = loss

        
        if not isinstance(x, list):
               x = [x]  # Convert single Value to a list
        if not isinstance(x, Value):
            x = [Value(xvalue) for xvalue in x]
            
        
        for epoch in range(epochs):
           output = self.forward(x)
 
           
           loss = loss_function(output, x)   
           loss.backward()
           
           self.update(lr)
           self.zero_grad()
           
           if epoch % 5 == 0:               
               print(f"Epoch: {epoch} \n Loss: {loss}")           
           
    
    def __repr__(self):
        return f"Total Layers: {len(self.layers)}"