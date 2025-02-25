import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

class Node():
    weights=[]
    #inputs=[]
    bias=0
    def __init__(self,weights:list,bias:float):
        self.weights=weights
        self.bias=bias
    def input(self,inputs:list):
        print(type(self))
        print(type(inputs))
        print(type(self.weights))
        self.__output=sigmoid(sum(map(lambda x,y:x*y,inputs,self.weights))+self.bias)
    def get_output(self)->float:
        return sigmoid(self.__output)

class InputNode(Node):
    __input=0
    def __init__(self):
        pass
    def intput(self,inputs:float):
        print("Hello world")
        self.__input=inputs
    def get_output(self):
        return self.__input

class OutputNode(Node):
    def get_output(self):
        return self.__output