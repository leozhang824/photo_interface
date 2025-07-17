import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def load_and_preprocess_data(start_datetime, filter_datetime, data):
    data['Time (seconds)'] = start_datetime + pd.to_timedelta(data['Time (seconds)'], unit='s')
    data['Time (seconds)'] = pd.to_datetime(data['Time (seconds)'])
    
    mask = data['Time (seconds)'] >= filter_datetime
    data = data.loc[mask]

    data['day'] = data['Time (seconds)'].dt.date
    data['time'] = data['Time (seconds)'].dt.hour * 3600 + data['Time (seconds)'].dt.minute * 60 + data['Time (seconds)'].dt.second

    
    # Identify feature columns (excluding 'Time (seconds)' and 'day')
    feature_columns = [col for col in data.columns if col not in ['Time (seconds)', 'day', 'voltage', 'time']]
    print("Feature columns:", feature_columns) 

    original_time = data['Time (seconds)']
    original_volt = data['voltage']
    # Normalize all features
    feature_scaler = MinMaxScaler()
    normalized_features = feature_scaler.fit_transform(data[feature_columns])

    # Normalize voltage separately
    voltage_scaler = MinMaxScaler()
    normalized_voltage = voltage_scaler.fit_transform(data[['voltage']])
    
    # Create a new DataFrame with normalized values
    normalized_df = pd.DataFrame(normalized_features, columns=feature_columns, index=data.index)
    normalized_df['voltage'] = normalized_voltage
    normalized_df['day'] = data['day']
    normalized_df['Time (seconds)'] = data['Time (seconds)']
    
    print(len(normalized_df))
    # Speed up training
    normalized_df = normalized_df.iloc[::2]

    
    return normalized_df, feature_scaler, feature_columns, voltage_scaler, original_volt, original_time

def prepare_sequences(data, sequence_length, feature_columns):
    sequences = []
    targets = []
    timestamps = []
    voltage_column = 'voltage'

    # Ensure 'day' column is datetime
    data['day'] = pd.to_datetime(data['day'])
    data = data.sort_values(['day', 'Time (seconds)'])

    # Group by day
    grouped = data.groupby('day')
    unique_days = list(grouped.groups.keys())

    print("Unique days in dataset:", unique_days)

    for i in range(len(unique_days) - 1):  # Stop one day before the last
        current_day = unique_days[i]
        next_day = unique_days[i+1]
        
        current_day_data = grouped.get_group(current_day)[feature_columns].values
        next_day_voltage = grouped.get_group(next_day)[voltage_column].values
        next_day_timestamp = grouped.get_group(next_day)['Time (seconds)'].values
        # Truncate or pad sequences to ensure consistent length
        current_day_data = current_day_data[:sequence_length]
        next_day_voltage = next_day_voltage[:sequence_length]
        next_day_timestamp = next_day_timestamp[:sequence_length]


        
        # Safety rail but this should always work
        if len(current_day_data) == sequence_length and len(next_day_voltage) == sequence_length:
            sequences.append(current_day_data)
            targets.append(next_day_voltage)
            timestamps.append(next_day_timestamp)
        else:
            print(f"Data lengths are messed up. Curr Day: {len(current_day_data)}, Next Day: {len(next_day_voltage)}")
            
    # for i in range(len(unique_days) - 1):  # Stop one before the last day
    #     current_day = unique_days[i]
    #     next_day = unique_days[i + 1]

    #     current_day_df = grouped.get_group(current_day)
    #     next_day_df = grouped.get_group(next_day)

    #     # Combine features into array
    #     current_day_data = current_day_df[feature_columns].values
    #     next_day_voltage = next_day_df[voltage_column].values
    #     next_day_timestamp = next_day_df['Time (seconds)'].values

    #     # Check how many full 12-hour sequences fit
    #     total_sequences = len(current_day_data) // sequence_length
    #     total_targets = len(next_day_voltage) // sequence_length
    #     total_chunks = min(total_sequences, total_targets)

    #     for chunk in range(total_chunks):
    #         start_idx = chunk * sequence_length
    #         end_idx = start_idx + sequence_length

    #         x_seq = current_day_data[start_idx:end_idx]
    #         y_seq = next_day_voltage[start_idx:end_idx]
    #         t_seq = next_day_timestamp[start_idx:end_idx]

    #         if len(x_seq) == sequence_length and len(y_seq) == sequence_length:
    #             sequences.append(x_seq)
    #             targets.append(y_seq)
    #             timestamps.append(t_seq)
    #         else:
    #             print(f"Incomplete chunk at {current_day}, chunk {chunk + 1}")
    
    return np.array(sequences), np.array(targets), np.array(timestamps)


def create_model(sequence_length, num_features):
    model = Sequential([
        LSTM(100, activation='tanh', input_shape=(sequence_length, num_features), 
             return_sequences=True, kernel_regularizer=l2(0.01)),
        LSTM(50, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01)),
        TimeDistributed(Dense(1))
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.005), loss='mae')
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    return model, early_stopping

def train_model(model, early_stopping, X_train, y_train, num_epochs, initial_epoch):

    model.fit(
        X_train, y_train, 
        epochs=num_epochs, 
        initial_epoch = initial_epoch,
        batch_size=16, 
        validation_split=0, 
        verbose=1,
        callbacks=[early_stopping]
    )
    
    return model

def make_pred_plot(start_datetime, filter_datetime, csv, min_epoch, max_epoch):
    sequence_length = 4*60

    # Load and preprocess the data
    data, feature_scaler, feature_columns, voltage_scaler, original_volt, original_time = load_and_preprocess_data(start_datetime, filter_datetime, csv)

    print(f"Data: {len(data)}")

    # Prepare sequences for LSTM
    X, y, timestamps = prepare_sequences(data, sequence_length, feature_columns)
    

    # Get the correct number of features
    num_features = X.shape[2]

    # Split the data into training and testing sets
    split = (len(X) - 1)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    timestamps_train, timestamps_test = timestamps[:split], timestamps[split:]

    print(type(X_train), X_train.shape)
    print(type(y_train), y_train.shape)


    print(f"X: {len(X)}")

    # Create and train the model
    model, early_stopping = create_model(sequence_length, num_features)

    figures = {}


    initial_epoch = 0
    for num_epochs in range(min_epoch, max_epoch, 10):
        model = train_model(model, early_stopping, X_train, y_train, num_epochs, initial_epoch)
        initial_epoch = num_epochs

        # Make predictions
        predictions = model.predict(X_test)

        # Inverse transform the predictions and actual values
        predictions_original = voltage_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
        y_test_original = voltage_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

        # After making predictions and inverse transforming
        y_test_flat = y_test_original.flatten()
        predictions_flat = predictions_original.flatten()

        # # Create a time-like x-axis (assuming each point represents a fixed time step)
        # time_steps = np.arange(len(y_test_flat))

        # Plot the results
        fig, ax = plt.subplots()
        ax.plot(original_time[::2], original_volt[::2], label="Original Data", color='lightgray', alpha=0.5)
        ax.plot(timestamps_test.flatten()[10:], y_test_original.flatten()[10:], label = "Actual")
        ax.plot(timestamps_test.flatten()[10:], predictions_original.flatten()[10:], label = "Preds")

        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Voltage")
        ax.set_title('LSTM Model: Actual vs Predicted Voltage for Next Day')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        figures[num_epochs] = fig

    return figures


    # fig 1 is 86 seq on 10 epoch
    # fig 2 is 864 seq on 50 epoch
    # testing in 12 hr chunks instead of 24 hr chunks right now because lack of data
    # seems to affect prediction model, need more data