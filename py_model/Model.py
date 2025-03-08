import Node
from random import random

def duplicate_structure(lst):
    from copy import deepcopy
    return replace_with_zero(deepcopy(lst))

def replace_with_zero(lst):
    """Recursively replace all values with 0"""
    if isinstance(lst, list):
        return [replace_with_zero(item) for item in lst]
    return 0  # Replace non-list elements with 0

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
            self.nodes[-1].append(Node.OutputNode([random()]*len(self.nodes[-2]),random()))
    def run(self,inputs:list):
        #input
        for i in range(len(self.nodes[0])):
            self.nodes[0][i].input(inputs[i])
        #run hidden layers
        for i in range(1,len(self.nodes)-1):
            for j in range(len(self.nodes[i])):
                self.nodes[i][j].input(list(map(lambda x:x.get_output(),self.nodes[i-1])))
                #print(list(map(lambda x:x.get_output(),self.nodes[i-1])))
                #print(self.nodes[i][j].get_output())
        #output
        for i in range(len(self.nodes[-1])):
            self.nodes[-1][i].input(list(map(lambda x:x.get_output(),self.nodes[-2])))
        return list(map(lambda x:x.get_output(),self.nodes[-1]))
    def get_error(self,answer):
        error=0
        for node in range(len(self.nodes[-1])):
            error+=(answer[node]-self.nodes[-1][node].get_output())**2
        return error
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
    def learn(self,teach_cases,teach_answers,test_cases,test_answers,step,cal_count):
        
        error=0
        for case in range(len(test_cases)):
            self.run(test_cases[case])
            error+=self.get_error(test_answers[case])
        print(f'Learning started by the initial error of {error}.')
        error_report=[error]
        model_data=self.get_model_data()
        for count in range(cal_count):
            gradient=duplicate_structure(self.get_model_data())
            error=0
            for case in range(len(teach_cases)):
                self.run(teach_cases[case])
                #output layer
                output_unit_errors=[0]*len(gradient[-1])
                for unit in range(len(gradient[-1])):
                    #print(self.nodes[-1][unit].get_raw_output(),self.nodes[-1][unit].get_output())
                    #print(self.nodes[-1][unit].get_output()*(1-self.nodes[-1][unit].get_output()))
                    error+=(teach_answers[case][unit]-self.nodes[-1][unit].get_output())**2
                    output_unit_errors[unit]=-(teach_answers[case][unit]-self.nodes[-1][unit].get_output())*self.nodes[-1][unit].get_output()*(1-self.nodes[-1][unit].get_output())
                    #print(output_unit_errors[unit])
                    if (type(gradient[-1][unit][0])==int):
                        pass
                    gradient[-1][unit][0]=list(map(lambda x,y:x.get_output()*output_unit_errors[unit]+y,self.nodes[-2],gradient[-1][unit][0]))#set the weight gradient
                    gradient[-1][unit][1]-=output_unit_errors[unit]#set the bias gradient
                #hidden layer
                for layer in range(len(gradient)-2,0,-1):
                    for unit in range(len(gradient[layer])):
                        unit_error=sum(map(lambda x,y:x*y[0][unit],output_unit_errors,model_data[layer+1]))*self.nodes[layer][unit].get_output()*(1-self.nodes[layer][unit].get_output())
                        
                        gradient[layer][unit][0]=list(map(lambda x,y:x.get_output()*unit_error+y,self.nodes[layer-1],gradient[layer][unit][0]))#set the weight gradient
                        gradient[layer][unit][1]-=unit_error
            #print(gradient)
            for layer in range(1,len(self.nodes)):
                for node in range(len(self.nodes[layer])):
                    data=self.nodes[layer][node].get_data()
                    data[0]=list(map(lambda x,y:x-y*step,data[0],gradient[layer][node][0]))#update weights
                    data[1]-=gradient[layer][node][1]*step
                    self.nodes[layer][node].load_data(data)
            error=0
            for case in range(len(test_cases)):
                self.run(test_cases[case])
                error+=self.get_error(test_answers[case])
            print(f'Learned {count+1} times, total error is {error}.')
            error_report.append(error)
        #self.run()
        
        print(f'Learning completed after {cal_count} times, total error is {error}.')
        return error_report