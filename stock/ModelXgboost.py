import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

data=pd.read_csv('stock/stock.csv',encoding='unicode_escape')
def get_data():
    return data
plt.plot(data['Close'])
plt.show()

#model=xgb.XGBRegressor()