import pandas as pd
import os

# function to draw datapoints in a specified timespan
def DrawDateData(filename, startDate, endDate):
    file = pd.read_excel(f"./data/{filename}.xlsx")
    file.sort_values(by='date', inplace=True)
    day = file.loc[(file['date'] >= startDate)  & (file['date'] <= endDate)]
    print(day.describe())
    day.plot(x='date', y='value', title=f"{filename}.xlsx", kind='line', figsize=(20, 10), style='.-')


# draw scatter plot of all the animals
for root, dirs, files in os.walk("./data/"):
    for file in files:
        print(file)
        temp = pd.read_excel(root+file)
        temp.columns = ["date", "value"]
        temp.sort_values(by='date', inplace=True)
        temp.plot(x='date', y='value', title=file, kind="scatter", figsize=(20, 10))


# draw barchart of the mean of every day of each animal
for root, dirs, files in os.walk("./data/"):
    for file in files:
        print(file)
        temp = pd.read_excel(root+file)
        temp.columns = ["date", "value"]
        groups = temp.groupby(pd.Grouper(key="date", axis=0, freq='1D')) #a df groupby object, frequency is 1 day
        temp_mean_grouped = groups.mean() # new df consist the mean of each day
        temp_mean_grouped = temp_mean_grouped.reset_index()
        temp_mean_grouped.sort_values(by='date', inplace=True)
        temp_mean_grouped.plot(x='date', y='value', title=file, kind="bar", figsize=(20, 10))

DrawDateData('1009', '2023-03-20', '2023-03-21')