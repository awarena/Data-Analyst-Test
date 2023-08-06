import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras import Sequential
from matplotlib import pyplot as plt

import os

cond_dict = {'1009.xlsx': ['2023-03-20'],
             '1224.xlsx': ['2023-02-05', '2023-02-27', '2023-02-28', '2023-03-24'],
             '1256.xlsx': ['2023-01-20', '2023-02-10', '2023-03-06'],
             '1261.xlsx': ['2023-01-25', '2023-02-18', '2023-02-19', '2023-03-14', '2023-03-15'],
             '1328.xlsx': ['2023-02-05', '2023-03-20']} # days with events for each animal

without_event = [] # list of time series data without events
without_event_date = [] # list of dates for time series data without events
with_event = [] # list of time series data with events
with_event_date = []    # list of dates for time series data with events

for root, dirs, files in os.walk("./data/"):
    for file in files:
        print(file)
        temp = pd.read_excel(f'./data/{file}')
        temp.columns = ["date", "value"]

        temp['date'] = temp['date'].dt.normalize() # remove time from date for grouping

        for day in temp['date'].unique():
            day_data = temp[temp['date'] == day][['value']].to_numpy() # get time series data for a day
            if file in cond_dict.keys() and str(day)[:10] in cond_dict[file]:
                with_event.append(day_data)
                with_event_date.append(day)
            else:
                without_event.append(day_data)
                without_event_date.append(day)

with_event = tf.keras.preprocessing.sequence.pad_sequences(with_event, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.0) # pad sequences to the same length
without_event = tf.keras.preprocessing.sequence.pad_sequences(without_event, maxlen=with_event.shape[1], dtype='float32', padding='post', truncating='post', value=0.0) # pad sequences to the same length

timesteps = with_event.shape[1] # number of timesteps
n_features = with_event.shape[2] # number of features

checkpoint_filepath = './checkpoint/{epoch:02d}-{loss:.4f}.keras'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    # save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(RepeatVector(timesteps))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')

model.fit(without_event, without_event, epochs=100, batch_size=32, verbose=1, callbacks=[model_checkpoint_callback]) # train model

model.save('auto_encoder_model_combined_training_mse.keras') # save model


model = tf.keras.models.load_model('auto_encoder_model_1feature_mse.keras')

with_event_pred = model.predict(with_event)
with_event_mse = [tf.keras.losses.mse(e, e_p).numpy().mean() for e, e_p in zip(with_event, with_event_pred)] # calculate the mean mse of a day that has event

without_event_pred = model.predict(without_event)
without_event_mse = [tf.keras.losses.mse(e, e_p).numpy().mean() for e, e_p in zip(without_event, without_event_pred)] # calculate the mean mse of a day that has no event

event_df = pd.DataFrame([[date, mse, label] for date, mse, label in zip(with_event_date, with_event_mse, [1 for i in range(len(with_event_date))])], columns=["date", "mse", "label"])
no_event_df = pd.DataFrame([[date, mse, label] for date, mse, label in zip(without_event_date, without_event_mse, [0 for i in range(len(without_event_date))])], columns=["date", "mse", "label"])

df = event_df.append(no_event_df, ignore_index=True)
df.plot.scatter(x='date', y='mse', c='label', title='Combined Training MSE', figsize=(20, 10), colormap='cool')
plt.legend(['event', 'no event'])
plt.savefig('combined_training_mse.png')