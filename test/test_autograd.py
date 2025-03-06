import unittest
import sys
sys.path.insert("/home/jv/GitHub/autograd/class")
from autograd.py import Value

class TestValue(unittest.TestCase):
    def setUp(self):
        self.ValueNode = Value(0.5)
    
    def testInit(self):
        self.assertEqual(self.ValueNode.value, 0.5)