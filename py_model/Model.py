import Node
from random import randint

class Model():
    nodes=[]#nodes[0]=input layer   nodes[len(nodes)-1]=output layer
    def __init__(self,inputs:int,hiddenLayerNodes:int,hiddenLayers:int,outputs:int):
        #spawn input layer
        self.nodes.append([])
        for _ in range(inputs):
            self.nodes[0].append(Node.InputNode())
        #spawn hidden layers
        for i in range(hiddenLayers):
            self.nodes.append([])
            for _ in range(1,hiddenLayerNodes+1):
                self.nodes[i].append(Node.Node([randint(0,1)]*len(self.nodes[i-1]),randint(0,1)))
        #spawn output layer
        self.nodes.append([])
        for _ in range(outputs):
            self.nodes[-1].append(Node.Node([randint(0,1)]*len(self.nodes[-2]),randint(0,1)))
    def run(self,inputs:list):
        #input
        for i in range(len(self.nodes[0])):
            print(type(self.nodes[0][i]))
            self.nodes[0][i].input(inputs[i])
        #run hidden layers
        for i in range(1,len(self.nodes)-1):
            for j in range(len(self.nodes[i])):
                self.nodes[i][j].input(list(map(lambda x:x.get_output()),self.nodes[i-1]))
        #output
        for i in range(len(self.nodes[-1])):
            self.nodes[-1][i].input(list(map(lambda x:x.get_output()),self.nodes[-2]))
        return list(map(lambda x:x.get_output(),self.nodes[-1]))