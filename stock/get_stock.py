import requests
import csv

class Stock():
    stock_num=0
    def __init__(self,stock_num):
        self.stock_num=stock_num
    def getStock(self):
        self.r = requests.post("https://www.cmoney.tw/api/internal/MobileService/ashx/GetDtnoData.ashx", data={"FilterNo": 0,"DtNo": "86489961","action": "getdtnodata","ParamStr":"AssignID=2330;CaptionMode=0;DTRange=1500;"})
        print(self.r.ok)
        print(self.r.json())
    def make_csv(self,filename='stock\\stock.csv'):
        with open(filename,'w',newline='') as file:
            data=self.r.json()
            writer=csv.writer(file)
            title=['Date','Open','High','Low','Close','Volume']
            writer.writerow(title)
            writer.writerows(map(lambda x:x[0:len(title)],data['Data']))


if __name__=="__main__":
    stock=Stock(0)
    stock.getStock()
    stock.make_csv()