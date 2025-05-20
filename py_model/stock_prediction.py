import Model
import pandas

model=Model.RecurrentModel(5,2,1,1)

stock_data=pandas.read_csv('stock\\stock.csv')
#print(stock_data.to_list())

teach_cases=[]
teach_answers=stock_data.loc[0][4]
for i in range(len(stock_data['Date'])-1,0,-1):
    teach_cases.append(stock_data.loc[i].to_list()[1:])
del stock_data
print ()

learn=True
if learn:
    learning=Model.RecurrentLearning(model)
    learning.learn([teach_cases],[[teach_answers]],cal_count=500)