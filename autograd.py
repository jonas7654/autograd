import math

class Value:
    def __init__(self, value : float):
        self.value = value
        self.gradient = 0
        self._backward = lambda : None
        self.leftChild = None
        self.rightChild = None
        self.operator= None
        
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
            
        result = Value(self.value + other.value)
        result.leftChild = self
        result.rightChild = other
        result.operator = '+'
        
        def backward():
            self.gradient += result.gradient 
            other.gradient += result.gradient 
        result._backward = backward
        
        return result
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
            
        result = Value(self.value * other.value)
        result.leftChild = self
        result.rightChild = other
        result.operator = "*"
        
        def backward():
            self.gradient += result.gradient * other.value
            other.gradient += result.gradient * self.value
        result._backward = backward
        
        return result
    
    def exp(self):
        result = Value(math.exp(self.value))
        result.leftChild = self
        result.rightChild = None
        result.operator = "exp"
        
        def backward():
            self.gradient += result.value * result.gradient
        result._backward = backward
        
        return result
    
    def ln(self):
        result = Value(math.log(self.value))
        result.leftChild = self
        result.rightChild = None
        result.operator = "ln"
        
        def backward():
            self.gradient += (1 / result.value) * result.gradient
        result._backward = backward
        
        return result
    
    def tanh(self):
        result = Value(math.tanh(self.value))
        result.leftChild = self
        result.rightChild = None
        result.operator = "tanh"
        
        def backward():
            self.gradient += (1 - self.value**2) * result.gradient
        result._backward = backward
        
        return result
            
    def backward(self):
        self.gradient = 1.0
        nodes_visited = set() # set's are implemented as hash tables which have expected lookup time of O(1)
        nodes_topo = []
        def topological_sort(node):
            if node not in nodes_visited:
                nodes_visited.add(node)
                if node.leftChild :
                    topological_sort(node.leftChild)
                if node.rightChild:
                    topological_sort(node.rightChild)
                nodes_topo.append(node)
    
        topological_sort(self)
        topo = list(reversed(nodes_topo)) # reversed returns an iterator
        
        for node in topo:
            node._backward()
            
    def zero_grad(self):
        self.gradient = 0
        if (self.leftChild):
            self.leftChild.zero_grad()
        if (self.rightChild):
            self.rightChild.zero_grad()
        
    
    def __repr__(self):
        return f"Node value: {self.value} \nNode gradient: {self.gradient} \n"
