import Node
from random import random

class Model():
    nodes=[]#nodes[0]=input layer   nodes[len(nodes)-1]=output layer
    def __init__(self,inputs:int,hiddenLayerNodes:int,hiddenLayers:int,outputs:int):
        #spawn input layer
        self.nodes.append([])
        for _ in range(inputs):
            self.nodes[0].append(Node.InputNode())
        #spawn hidden layers
        for i in range(1,hiddenLayers+1):
            self.nodes.append([])
            for _ in range(hiddenLayerNodes):
                self.nodes[i].append(Node.Node([random()]*len(self.nodes[i-1]),random()))
        #spawn output layer
        self.nodes.append([])
        for _ in range(outputs):
            self.nodes[-1].append(Node.Node([random()]*len(self.nodes[-2]),random()))
    def run(self,inputs:list):
        #input
        for i in range(len(self.nodes[0])):
            self.nodes[0][i].input(inputs[i])
        #run hidden layers
        for i in range(1,len(self.nodes)-1):
            for j in range(len(self.nodes[i])):
                self.nodes[i][j].input(list(map(lambda x:x.get_output(),self.nodes[i-1])))
                print(self.nodes[i][j].get_output())
        #output
        for i in range(len(self.nodes[-1])):
            self.nodes[-1][i].input(list(map(lambda x:x.get_output(),self.nodes[-2])))
        return list(map(lambda x:x.get_output(),self.nodes[-1]))
    def get_model_data(self):
        output=[[]]
        for i in range(1,len(self.nodes)):
            output.append([])
            for j in range(len(self.nodes[i])):
                output[i].append(self.nodes[i][j].get_data())
        return output
    def load_data(self,data:list):
        for i in range(1,len(self.nodes)):
            for j in range(len(self.nodes[i])):
                self.nodes[i][j].load_data(data[i][j])