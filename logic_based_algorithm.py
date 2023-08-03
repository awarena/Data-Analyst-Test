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
# min event gap decides the minimum day between each events
def mean_based_anomaly_estimation(animal, threshold_factor=3.0, min_event_gap=20):
    df = pd.read_excel(f"./data/{animal}.xlsx")
    df.columns = ["date", "value"]
    df = df.sort_values(by='date')
    df['date'] = pd.to_datetime(df['date'])

    df['date'] = df['date'].dt.normalize() # truncate the time part of the object
    overall_mean_value = df['value'].mean() # overall mean
    df['mean_value'] = df.groupby('date')['value'].transform('mean') # add another column for mean value of each day
    df['is_anomaly'] = df['value'] > (threshold_factor * overall_mean_value) # add another column to indicate if the day is anomaly or not
    
    if not (True in df['is_anomaly'].unique()):
        print(f"Animal {animal} does not have any event.\nPlease note this is only for reference. A manual validation is needed for every projection the algorithm makes.\n")
        return df
    
    # apply the constraint of a minimum gap between events
    filtered_events = []

    prev_event_date = None
    for index, row in df.iterrows():
        if row['is_anomaly']:
            event_date = row['date']
            if prev_event_date is None or (event_date - prev_event_date).days > min_event_gap:
                filtered_events.append(index)
                prev_event_date = event_date
                print(f"An event might happen on {event_date.date()} of animal {animal}.\nPlease note this is only for reference. A manual validation is needed for every projection the algorithm makes.\n")


    df['is_anomaly'] = False
    df.loc[filtered_events, 'is_anomaly'] = True

    return df

print(mean_based_anomaly_estimation('1356', 4.0))
