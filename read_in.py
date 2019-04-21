import pandas as pd

from pandas import ExcelWriter
from pandas import ExcelFile

data = pd.read_excel('formatted_data.xlsx', usecols=[2,3,4,5,6,7,8])
numdata = pd.read_excel('formatted_vals.xlsx')

datatrain = numdata.loc[2:62].to_numpy() 
datatest = numdata.loc[63:84].to_numpy()

