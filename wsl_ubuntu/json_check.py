import pandas as pd

df = pd.read_json("output/batch_test_output.json") 
print(df.columns.to_list())

des = df['embedding'].iloc[1]
print(len(des))

