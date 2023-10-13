import pandas as pd
a = pd.read_csv('not_divided_by_C.csv')
b = pd.read_csv('divided_by_C.csv')
print("RD without Normalize")
print(a.iloc[:,1].describe())
print("RD with Normalize")
print(b.iloc[:,1].describe())