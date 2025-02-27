from Model import Model

model=Model(1,1,1,1)
model.load_data([[],[([5],3)],[([1],0)]])
print(model.run([9]))
print(model.get_model_data())