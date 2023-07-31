import pandas as pd
import os


temp = pd.read_excel("./data/1261.xlsx")
day = temp.loc[(temp['date'] >= '2023-02-18')  & (temp['date'] <= '2023-02-19')]
print(day.describe())
day.plot(x='date', y='value', title="1261.xlsx", kind="scatter", figsize=(20, 10))


for root, dirs, files in os.walk("./data/"):
    for file in files:
        print(file)
        temp = pd.read_excel(root+file)
        temp.columns = ["date", "value"]
        temp.plot(x='date', y='value', title=file, kind="scatter", figsize=(20, 10))



for root, dirs, files in os.walk("./data/"):
    for file in files:
        print(file)
        temp = pd.read_excel(root+file)
        temp.columns = ["date", "value"]
        groups = temp.groupby(pd.Grouper(key="date", axis=0, freq='1D'))
        temp_count_grouped = groups.mean()
        temp_count_grouped = temp_count_grouped.reset_index()
        temp_count_grouped.plot(x='date', y='value', title=file, kind="bar", figsize=(20, 10))