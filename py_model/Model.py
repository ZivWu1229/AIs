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

def sigmoid_prime(sigmoid):
    return sigmoid*(1-sigmoid)

class Model():
    nodes=[]#nodes[0]=input layer   nodes[-1]=output layer
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
    def input(self,inputs:list):
        #input
        for i in range(len(self.nodes[0])):
            if type(self.nodes[0][i])==Node.RecurrentNode:
                break
            self.nodes[0][i].input(inputs[i])
    def hidden(self):
        #run hidden layers
        for i in range(1,len(self.nodes)-1):
            for j in range(len(self.nodes[i])):
                self.nodes[i][j].input(list(map(lambda x:x.get_output(),self.nodes[i-1])))
    def output(self):
        #output
        for i in range(len(self.nodes[-1])):
            self.nodes[-1][i].input(list(map(lambda x:x.get_output(),self.nodes[-2])))
        return list(map(lambda x:x.get_output(),self.nodes[-1]))
    def run(self,inputs):
        self.input(inputs)
        self.hidden()
        self.output()
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

    #Machine Learning Part
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
            yield [error,self.get_model_data()]
            error_report.append(error)
        #self.run()
        
        print(f'Learning completed after {cal_count} times, total error is {error}.')
        yield -1
        #return error_report

class BasicModel(Model):
    pass

class RecurrentModel(Model):
    def __init__(self, inputs, hiddenLayerNodes, hiddenLayers, outputs):
        super().__init__(inputs, hiddenLayerNodes, hiddenLayers, outputs)
        for _ in range(hiddenLayerNodes):
            self.nodes[0].extend([Node.RecurrentNode()])
    def hidden(self):
        recurrent_index=len(self.nodes[-2])
        #run hidden layers
        for i in range(1,len(self.nodes)-1):
            for j in range(len(self.nodes[i])):
                if i==1:
                    input=list(map(lambda x:x.get_output(),self.nodes[0][:-recurrent_index]))
                    input.append(self.nodes[0][-recurrent_index+j].get_output())
                else:
                    input=list(map(lambda x:x.get_output(),self.nodes[i-1]))
                self.nodes[i][j].input(input)
    def refresh(self):
        for i in range(len(self.nodes[0])):
            if type(self.nodes[0][i])==Node.RecurrentNode:
                self.nodes[0][i].input(0)
    def run(self,input:list):
        self.refresh()
        while True:
            super().input(input)
            self.hidden()
            results=list(map(lambda x:x.get_output(),self.nodes[-2]))#get hidden layer outputs for next loop
            super().output()
            input = yield list(map(lambda x:x.get_output(),self.nodes[-1]))
            if input==0:
                #print(results)
                input=yield results
            if type(input)!=list:
                break
            for i in range(len(results)):
                self.nodes[0][-len(self.nodes[-2])+i].input(results[i])
            #map(lambda node,result:node.input(result),self.nodes[0][-len(self.nodes[-2]):],results)#input the result to recurrent node
        super().output()
    
    def learn(self, teach_cases, teach_answers=[], test_cases=[], test_answers=[], step=0.05, cal_count=1):
        #get all recurrent input
        
        for case in range(1,len(teach_cases)):
            g=self.run(teach_cases[case][0])
            next(g)
            recurrentInputs=[]
            for inputNum in range(len(teach_cases[case])):
                g.send(0)
                recurrentInputs.append(list(map(lambda x:x.get_output(),self.nodes[-2])))
                g.send(teach_cases[case][inputNum])
            #print(recurrentInputs)


class RecurrentLearning():
    def __init__(self,model:RecurrentModel):
        self.model=model
    def learn(self,teach_cases,teach_answers):
        gradient=duplicate_structure(self.model.get_model_data())
        for case in range(len(teach_cases)):
            #get recurrent node value
            g=self.model.run(teach_cases[case][0])
            next(g)
            recurrentInputs=[]
            for inputNum in range(len(teach_cases[case])):
                g.send(0)
                recurrentInputs.append(list(map(lambda x:x.get_output(),self.model.nodes[-2])))
                g.send(teach_cases[case][inputNum])
            #get unit error of output
            output_unit_error=[0]*len(self.model.nodes[-1])
            #set output layer gradient
            for unit in range(len(self.model.nodes[-1])):
                #get output unit error
                output_unit_error[unit]=-2*(teach_answers[case][unit]-self.model.nodes[-1][unit].get_output())*sigmoid_prime(self.model.nodes[-1][unit].get_output())
                #set gradient
                gradient[-1][unit][0]=list(map(lambda i:output_unit_error[unit]*recurrentInputs[0][i]+gradient[-1][unit][0][i],list(range(len(self.model.nodes[-2])))))
                gradient[-1][unit][1]-=output_unit_error[unit]#set the bias gradient
            #set hidden layer gradient
            print(gradient)