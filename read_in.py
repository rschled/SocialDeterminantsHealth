import pandas as pd

from pandas import ExcelWriter
from pandas import ExcelFile

data = pd.read_excel('formatted_data.xlsx', usecols=[2,3,4,5,6,7,8])
print(data.columns)
print(data.size)
print(data.loc[1])
print(data.head())

