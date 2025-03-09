import tkinter
import ttkbootstrap as tb
import os.path,json
from Model import Model

#set up model
if os.path.isfile('d:\\code\\python\\AIs\\py_model\\alphabet\\model_data.json'):
    with open('d:\\code\\python\\AIs\\py_model\\alphabet\\model_data.csv','r') as file:
        model_data=json.loads(file.read())

model=Model(20,3,1,4)
model.load_data(model_data)

#set up gui

win=tkinter.Tk()

def submit():
    result=model.run(list(map(lambda x:x.get(),status)))
    ans=0
    for i in range(len(result)):
        if result[i]>result[ans]:
            ans=i
    alphabets=['A','P','L','E']
    print(result)
    textVar.set(alphabets[ans])
    possibility.set(str(int(result[ans]*100))+'%')

def on_click():
    #print(list(map(lambda x:x.get(),status)))
    pass

status =[]
buttons=[]
textVar=tkinter.StringVar()
possibility=tkinter.StringVar()
for i in range(5):
    for j in range(4):
        status.append(tkinter.IntVar())
        buttons.append(tb.Checkbutton(bootstyle="danger, toolbutton, outline",
                                      text=len(buttons),
                                      variable=status[-1],
                                      onvalue=1,
                                      offvalue=0,
                                      command=submit,
                                      width=3))
        buttons[-1].grid(column=j,row=i)
#tkinter.Button(win,text="Submit",command=submit).grid(row=5,column=0)
tkinter.Label(win,textvariable=textVar,font=('Arial',100,'bold')).grid(column=0,row=5,columnspan=5)
tkinter.Label(win,textvariable=possibility,font=('Arial',50)).grid(column=0,row=6,columnspan=5)
win.mainloop()