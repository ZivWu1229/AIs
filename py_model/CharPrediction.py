from Model import *
import pytest

model=RecurrentModel(3,2,1,3)

H1=[[3.02,18.29,2.53,65.06],16.88]
H2=[[7.92,8.52,21.55,4.91],13.44]

Z1=[[72.83,90.10],96.13]
Z2=[[-66.62,-4.67],-12.99]
Z3=[[10.98,-74.34],-5.70]

model.load_data([[],[H1,H2],[Z1,Z2,Z3]])

#print(model.run([[0,1,0]]))

test_cases=[
    [[1,0,0],[0,1,0]],
    [[1,0,0],[0,0,1]],
    [[0,1,0],[1,0,0]],
    [[0,1,0],[0,0,1]],
    [[0,0,1],[1,0,0]],
    [[0,0,1],[0,1,0]],
    [[0,0,1]],
    [[0,1,0]]
]
answers=[
    [0,0,1],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [0,1,0],
    [1,0,0],
    [0,1,0],
    [0,0,1]
]
def test():
    for i in range(len(test_cases)):
        result=model.run(test_cases[i])
        print(f'Result:{list(map(lambda x:round(x,1),result))}')
        assert sum(map(lambda x,y:(x-y)<0.1,answers[i],result))==3


if __name__=='__main__':
    learn=True
    #ML
    if learn:
        learning=RecurrentLearning(model)
        learning.learn(test_cases,answers)
    else:
    # model.learn(test_cases)
        for i in range(len(test_cases)):
            g=model.run(test_cases[0][0])
            next(g)
            result=[]
            for j in range(len(test_cases[i])):
                result=g.send(test_cases[i][j])
            print(result)
