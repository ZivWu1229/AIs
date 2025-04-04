from Model import BasicModel
import csv,json,os.path

model=BasicModel(20,3,1,4)


H1=([\
0.05,0.00,0.00,0.00,\
0.00,0.09,0.00,0.00,\
0.00,0.00,0.00,0.00,\
0.00,0.31,0.00,0.30,\
0.00,0.74,7.67,0.41],7.59)
H2=([\
0.05,1.98,2.96,0.50,\
0.00,0.00,0.00,1.43,\
0.73,1.55,1.20,0.30,\
0.67,0.00,0.80,0.00,\
0.10,0.05,0.00,0.00],9.07)
H3=([\
0.00,0.00,0.00,0.00,\
0.00,0.36,2.58,0.02,\
0.00,0.00,2.67,1.45,\
0.00,0.12,3.28,4.54,\
0.00,0.00,0.00,0.18],6.67)

Z1=([ 0.00, 0.49,16.87],10.16)
Z2=([ 0.00,39.86, 0.00],30.10)
Z3=([33.99, 0.00, 0.00],26.83)
Z4=([34.00,34.48, 0.00],34.43)

test_case=list("""
0	1	1	0
1	0	0	1
1	1	1	0
1	0	0	0
1	0	0	0
""")
i=0
while len(test_case)>i:
        if test_case[i]=='\t'or test_case[i]=='\n':
            test_case.pop(i)
        else:
            test_case[i]=int(test_case[i])
            i+=1

del i
#print(test_case)

model.load_data([[],[H1,H2,H3],[Z1,Z2,Z3,Z4]])
#model.run(test_case)

#print(model.run(test_case))

def convert_to_int(lst):
    if isinstance(lst, list):
        return [convert_to_int(item) for item in lst]
    return int(lst)  # Replace non-list elements with 0

test_case=[]
answer=[]
model_data=[]

with open('py_model\\alphabet\\chr_img.csv','r') as file:
    csvFile=csv.reader(file)
    for line in csvFile:
        test_case.append(list(map(lambda x:int(x),line[1:])))

with open('py_model\\alphabet\\teacher.csv','r') as file:
    csvFile=csv.reader(file)
    for line in csvFile:
        answer.append(list(map(lambda x:int(x),line)))



import matplotlib.pyplot as plt
half=len(test_case)//2
learning_report=[]
min_point=0
g=model.learn(test_case[:half],answer[:half],test_case[half:],answer[half:],0.01,5000)
while True:
    
    report=next(g)
    if report==-1:
        break
    learning_report.append(report)
    if report[0]<learning_report[min_point][0]:
        min_point=len(learning_report)-1
plt.plot(list(map(lambda x:x[0],learning_report)))


print(f'Choosing the data with the lowest error of {learning_report[min_point][0]} for the final result.')


with open('py_model\\alphabet\\model_data.json', 'w', newline='') as file:
    file.write(json.dumps(learning_report[min_point][1]))
plt.show()
