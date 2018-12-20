import pandas as pd
import numpy as np
from pandas import DataFrame

gender = []
X = []

df = pd.read_excel (r'C:\Users\Golam Rabbani\Downloads\500-person-gender-height-weight-bodymassindex\500_Person_Gender_Height_Weight_Index.xlsx')

df = df.values

df = np.asarray(df)

df = np.split(df, 500)


for i in range(0, len(df)):
    prime = []
    
    arr = np.split(df[i], 1)
    arr = np.asarray(arr)
    arr = arr[0][0]
    arr = np.split(arr, 3)
    gender.append(str(arr[0][0]))
    arr = arr[1:3]
    prime.append(arr[0][0])
    prime.append(arr[1][0])
    X.append(prime)

print(gender)
print(X)
