import pandas as pd
import numpy as np

# def EstimateEventDate(animal, threshold_percent=0.1, value_threshold = 0.90):
#     eventDates = []
#     data = pd.read_excel(f"./data/{animal}.xlsx")
#     data.columns = ["date", "value"]
#     data = data.sort_values(by='date')
#     data['date'] = data['date'].dt.date
#     data['date'] = pd.to_datetime(data['date'])

#     grouped_by_date = data.groupby('date')
#     for date, group in grouped_by_date:
#         values_for_day = group['value'].values
#         num_values_for_day = len(values_for_day)
#         threshold_count = int(num_values_for_day * threshold_percent)
#         percentile = np.percentile(values_for_day, value_threshold * 100)

#         num_values_larger_than_percentile = sum(value > percentile for value in values_for_day)

#         if num_values_larger_than_percentile > threshold_count:
#             eventDates.append(date)
    
#     if len(eventDates) != 0:
#         for date in eventDates:
#             print(f'An event might happen on day {date} of animal {animal}.\nPlease note this is only for reference, please validate your analysis on the data manually.\n')
#     else:
#         print(f'Seems like animal {animal} does not have any event.\nPlease note this is only for reference, please validate your analysis on the data manually.\n')
    

# EstimateEventDate('1224')



# animal is the filename
# threshold factor decides the sensitivity of the algorithm (the higher, the more sensitive)
# min outliers decides the minimum outliers a day to constitute an event
def mean_based_anomaly_estimation(animal, threshold_factor=2.5, min_outlier=10):
    df = pd.read_excel(f"./data/{animal}.xlsx")
    df.columns = ["date", "value"]
    df = df.sort_values(by='date')
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.normalize()  # truncate the time part of the object
    overall_mean_value = df['value'].mean()  # overall mean
    df['is_anomaly'] = df['value'] > (threshold_factor * overall_mean_value)  # add another column to indicate if the day is an anomaly or not

    if not df['is_anomaly'].any():
        print(f"Animal {animal} does not have any event.\nPlease note this is only for reference. A manual validation is needed for every projection the algorithm makes.\n")
        return df

    # count the number of outliers for each day
    df['outlier_count'] = df.groupby('date')['is_anomaly'].transform('sum')

    # apply the constraint of a minimum number of outliers in a day
    event_days = pd.to_datetime(df[df['outlier_count'] >= min_outlier]['date'].unique())
    
    if event_days.size == 0:
        print(f"No event days found for animal {animal} with a minimum of {min_outlier} outliers per day.\nPlease note this is only for reference. A manual validation is needed for every projection the algorithm makes.\n")
    
    for day in event_days:
        print(f"An event might happen on {day.date()} of animal {animal}.\nPlease note this is only for reference. A manual validation is needed for every projection the algorithm makes.\n")  

    df['is_anomaly'] = False
    df.loc[df['date'].isin(event_days), 'is_anomaly'] = True

    return df

# example usage
print(mean_based_anomaly_estimation('1009'))
print(mean_based_anomaly_estimation('1224'))
print(mean_based_anomaly_estimation('1013'))
print(mean_based_anomaly_estimation('1215'))
print(mean_based_anomaly_estimation('1356'))
print(mean_based_anomaly_estimation('1256'))
