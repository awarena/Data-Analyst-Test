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
             '1328.xlsx': ['2023-02-05', '2023-03-20']} # days with events

for root, dirs, files in os.walk("./data/"):
    for file in files: 
        without_event = [] # data without event
        without_event_date = [] # date without event
        with_event = [] # data with event
        with_event_date = [] # date with event

        temp = pd.read_excel(f'./data/{file}')
        temp.columns = ["date", "value"]

        temp['date'] = temp['date'].dt.normalize() # remove time


        if file in cond_dict.keys():
            for day in temp['date'].unique():
                day_data = temp[temp['date'] == day][['value']].to_numpy() # get data for each day
                if file in cond_dict.keys() and str(day)[:10] in cond_dict[file]: 
                    with_event.append(day_data)
                    with_event_date.append(day)
                else:
                    without_event.append(day_data)
                    without_event_date.append(day)
        else:
            unique_dates = temp['date'].unique() # get unique dates
            test_date = np.random.choice(unique_dates, int(unique_dates.shape[0]*0.1), replace=False) # randomly select 10% of data as test data
            for day in temp['date'].unique():
                day_data = temp[temp['date'] == day][['value']].to_numpy()
                if day in test_date: # if day is in test data
                    with_event.append(day_data) # append to with_event. This will be used as test data
                    with_event_date.append(day)
                else:
                    without_event.append(day_data)
                    without_event_date.append(day)
        
        with_event = tf.keras.preprocessing.sequence.pad_sequences(with_event, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.0) # pad sequences
        without_event = tf.keras.preprocessing.sequence.pad_sequences(without_event, maxlen=with_event.shape[1], dtype='float32', padding='post', truncating='post', value=0.0) # pad sequences

        timesteps = with_event.shape[1] # get timesteps
        n_features = with_event.shape[2] # get n_features

        checkpoint_filepath = f'./checkpoint/{file[:-5]}/' + '{epoch:02d}-{loss:.4f}.keras'
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

        model.fit(without_event, without_event, epochs=100, batch_size=32, verbose=1, callbacks=[model_checkpoint_callback])



        with_event_pred = model.predict(with_event)
        with_event_mse = [tf.keras.losses.mse(e, e_p).numpy().mean() for e, e_p in zip(with_event, with_event_pred)]  # get mse mean for each day

        without_event_pred = model.predict(without_event)
        without_event_mse = [tf.keras.losses.mse(e, e_p).numpy().mean() for e, e_p in zip(without_event, without_event_pred)] # get mse mean for each day


        event_df = pd.DataFrame([[date, mse, label] for date, mse, label in zip(with_event_date, with_event_mse, [1 for i in range(len(with_event_date))])], columns=["date", "mse", "label"]) # create dataframe for event
        no_event_df = pd.DataFrame([[date, mse, label] for date, mse, label in zip(without_event_date, without_event_mse, [0 for i in range(len(without_event_date))])], columns=["date", "mse", "label"]) # create dataframe for no event

        df = event_df.append(no_event_df, ignore_index=True)
        if file in cond_dict.keys():
            title = f'{file} with event'
        else:
            title = f'{file} without event'


        import pandas as pd
        import matplotlib.pyplot as plt


        # Set the figure size
        plt.figure(figsize=(20, 10))

        # Create the scatter plot for df and store the ax object
        ax = plt.gca()

        # Plot the data for 'df'
        df_plot = ax.scatter(x=df['date'], y=df['mse'], c=df['label'], vmin=0, vmax=1, cmap='cool', label='test')

        # # Plot the data for 'no_event_df'
        no_event_plot = ax.scatter(x=no_event_df['date'], y=no_event_df['mse'], c=no_event_df['label'], vmin=0, vmax=1, cmap='cool', label='no event')

        # Get the handles and labels from the scatter plots
        handles, labels = ax.get_legend_handles_labels()

        # Modify the labels based on the condition
        modified_labels = ['event' if file in cond_dict.keys() else 'test no event', 'no event']

        # Update the legend with the modified labels
        ax.legend(handles, modified_labels)

        # Set labels for axes (optional)
        ax.set_xlabel('Date')
        ax.set_ylabel('MSE')

        # Set the title
        ax.set_title(title)

        if not os.path.exists('./result/'):
            os.makedirs('./result/')
        plt.savefig(f'./result/{file[:-5]}.png')
