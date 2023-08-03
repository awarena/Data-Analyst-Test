import pandas as pd
import os

# function to draw datapoints in a specified timespan
def DrawDateData(filename, startDate, endDate):
    file = pd.read_excel(f"./data/{filename}.xlsx")
    file.sort_values(by='date', inplace=True)
    file.columns = ["date", "value"]
    day = file.loc[(file['date'] >= startDate)  & (file['date'] <= endDate)]
    print(day.describe())
    day.plot(x='date', y='value', title=f"{filename}.xlsx - {startDate}", kind='line', figsize=(20, 10), style='.-')


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

# Draw specified date of 1009
DrawDateData('1009', '2023-03-20', '2023-03-21')

# Draw specified date of 1224
DrawDateData('1224', '2023-02-05', '2023-02-06')
DrawDateData('1224', '2023-02-27', '2023-02-28')
DrawDateData('1224', '2023-02-28', '2023-03-01')
DrawDateData('1224', '2023-03-24', '2023-03-25')

# Draw specified date of 1256
DrawDateData('1256', '2023-01-20', '2023-01-21')
DrawDateData('1256', '2023-02-10', '2023-02-11')
DrawDateData('1256', '2023-03-06', '2023-03-07')

# Draw specified date of 1261
DrawDateData('1261', '2023-01-25', '2023-01-26')
DrawDateData('1261', '2023-02-18', '2023-02-19')
DrawDateData('1261', '2023-02-19', '2023-02-20')
DrawDateData('1261', '2023-03-14', '2023-03-15')
DrawDateData('1261', '2023-03-15', '2023-03-16')

# Draw specified date of 1328
DrawDateData('1328', '2023-02-05', '2023-02-06')
DrawDateData('1328', '2023-03-20', '2023-03-21')