import math

class Value:
    def __init__(self, _value : float):
        self.value = _value
        self.gradient = 0
        self._backward = lambda : None
        self.leftChild = None
        self.rightChild = None
        self.operator= None
        
    def __add__(self, other):
        result = Value(self.value + other.value)
        result.leftChild = self
        result.rightChild = other
        result.operator = '+'
        
        def backward():
            self.gradient = result.gradient 
            other.gradient = result.gradient 
        result._backward = backward
        
        return result
    
    def __mul__(self, other):
        result = Value(self.value * other.value)
        result.leftChild = self
        result.rightChild = other
        result.operator = "*"
        
        def backward():
            self.gradient += result.gradient * other._value
            other.gradient += result.gradient * self._value
        result._backward = backward
        
        return result
    
    def exp(self):
        result = Value(math.exp(self.value))
        result.leftChild = self
        result.rightChild = None
        result.operator = "exp"
        
        def backward():
            self.gradient = result.value * result.gradient
        result._backward = backward
        
        return result
    
    def ln(self):
        result = Value(math.log(self.value))
        result.leftChild = self
        result.rightChild = None
        result.operator = "ln"
        
        def backward():
            self.gradient = (1 / result.value) * result.gradient
        result._backward = backward
        
        return result
            
    def backward(self):
        self._backward()
        if (self.leftChild):
            self.leftChild.backward()
        if (self.rightChild):
            self.rightChild.backward()
        
    
    def __repr__(self):
        return f"Node value: {self.value} \nNode gradient: {self.gradient} \n"
