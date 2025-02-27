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
        self.__output=sum(map(lambda x,y:x*y,inputs,self.weights))+self.bias
    def get_output(self)->float:
        return self.__output
        #return sigmoid(self.__output)
    def get_data(self)->tuple:
        return (self.weights,self.bias)
    def load_data(self,data:tuple):
        self.weights=data[0]
        self.bias=data[1]

class InputNode(Node):
    __input=0
    def __init__(self):
        pass
    def input(self,inputs:float):
        self.__input=inputs
    def get_output(self):
        return self.__input

class OutputNode(Node):
    def get_output(self):
        return self.__output