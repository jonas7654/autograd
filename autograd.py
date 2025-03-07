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
    
    def __radd__(self, other):
        """Handles int/float + Value (calls __add__)"""
        return self.__add__(other)  # This makes 5 + Value(3) work!

    
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

    def __rmul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other.__mul__(self)
    
    def __pow__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
            
        result = Value(self.value ** other.value)
        result.leftChild = self
        result.rightChild = other
        result.operator = "pow"
        
        def backward():
            self.gradient += result.gradient * (other.value * (self.value **(other.value - 1)))
            if self.value <= 0:
                self.value = 1e-10
            other.gradient += result.gradient * ((self.value ** other.value) * math.log(self.value))
        result._backward = backward
        
        return result
    
    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
            
        result = Value(self.value - other.value)
        result.leftChild = self
        result.rightChild = other
        result.operator = "-"
        
        def backward():
            self.gradient += result.gradient * 1.0
            other.gradient += result.gradient * (-1.0)
        result._backward = backward
        
        return result
    
    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.value / other.value)
        result.leftChild = self
        result.rightChild = other
        result.operator = "/"
        
        def backward():
            self.gradient += result.gradient * (1 / other.value)  # ∂(a/b) / ∂a = 1/b
            other.gradient += result.gradient * (-(self.value) / (other.value ** 2))  # ∂(a/b) / ∂b = -a / b²
        result._backward = backward
        
        return result
    
    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other.__truediv__(self)
    
    # Define unary minus
    def __neg__(self):
        return Value(-self.value)
    
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
    
    def sigmoid(self):
        result = Value(math.exp(self.value) / (1 + math.exp(self.value)))
        result.leftChild = self
        result.rightChild = None
        result.operator = "sigmoid"
        def backward():
            derivative_sigmoid = result.value * (1 - result.value)
            self.gradient += result.gradient * derivative_sigmoid
        result._backward = backward
        
        return result
                
    def backward(self):
        """
        We want to begin at the current Node so we set the gradient of it to:
        df/df = 1.0

        Furthermore we need to apply a topological ordering to the computational graph in order to
        deal with things like this:
        
        node1 = a + b
        node2 = node1 * c
        node3 = node2 * a
        
        where a has an edge to node1 but also to node 2
        """
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
        return f"Value({self.value})"
