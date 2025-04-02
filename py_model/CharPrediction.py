from Model import RecurrentModel

model=RecurrentModel(3,2,1,3)

H1=[[3.02,18.29,2.53,65.06],16.88]
H2=[[7.92,8.52,21.55,4.91],13.44]

Z1=[[72.83,90.10],96.13]
Z2=[[-66.62,-4.67],-12.99]
Z3=[[10.98,-74.34],-5.70]

model.load_data([[],[H1,H2],[Z1,Z2,Z3]])

print(model.run([[0,1,0],[1,0,0]]))
