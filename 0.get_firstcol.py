import pandas as pd
import numpy as np

base = pd.read_csv("BACE-1_dataset/bace.csv")
# print(base.head())

first_col = base.iloc[:,0]
print(first_col)
first_col.to_csv("BACE-1_dataset/first_col.csv", index=False)