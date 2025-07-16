import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, RepeatVector
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt



### IMPORTANT
# Current iteration does the following:
#       Is running on weird 10/7 data
#       Mess with batch size -- overlapping sequences?
#       mess with model layers
#       df['hour_sin'] = np.sin(2 * np.pi * (hour + minute/60)/24)
#       Trained to predict voltage for the following 24 hour period
# Need to fix these

def load_and_preprocess_data(data):
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['day'] = data['Timestamp'].dt.date
    
    # Drop specified columns
    columns_to_drop = ['Dev1 Channel0', 'Dev1 Channel1', 'Dev1 Channel2', 'Soil Moisture', 'Soil Temp', 'Electrical Conductivity']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    # Identify feature columns (excluding 'Timestamp' and 'day')
    feature_columns = [col for col in data.columns if col not in ['Timestamp', 'day', 'Dev1 Channel3']]
    print("Feature columns:", feature_columns) 

    original_time = data['Timestamp']
    original_volt = data['Dev1 Channel3']
    # Normalize all features
    feature_scaler = MinMaxScaler()
    normalized_features = feature_scaler.fit_transform(data[feature_columns])

    # Normalize voltage separately
    voltage_scaler = MinMaxScaler()
    normalized_voltage = voltage_scaler.fit_transform(data[['Dev1 Channel3']])
    
    # Create a new DataFrame with normalized values
    normalized_df = pd.DataFrame(normalized_features, columns=feature_columns)
    normalized_df['Dev1 Channel3'] = normalized_voltage
    normalized_df['day'] = data['day']
    normalized_df['Timestamp'] = data['Timestamp']
    
    # Speed up training
    normalized_df = normalized_df.iloc[::100]
    
    return normalized_df, feature_scaler, feature_columns, voltage_scaler, original_volt, original_time

def prepare_sequences(data, sequence_length, feature_columns):
    sequences = []
    targets = []
    voltage_column = 'Dev1 Channel3'
    timestamps = []

    # Convert 'day' to datetime if it's not already
    data['day'] = pd.to_datetime(data['day'])    
    # Sort the data by day
    data = data.sort_values(['day', 'Timestamp'])
    
    # Group the data by day
    grouped = data.groupby('day')
    
    # Get unique days
    unique_days = list(grouped.groups.keys())
    
    for i in range(len(unique_days) - 1):  # Stop one day before the last
        current_day = unique_days[i]
        next_day = unique_days[i+1]
        
        current_day_data = grouped.get_group(current_day)[feature_columns].values
        next_day_voltage = grouped.get_group(next_day)[voltage_column].values
        next_day_timestamp = grouped.get_group(next_day)['Timestamp'].values
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
    
    return np.array(sequences), np.array(targets), np.array(timestamps)

def create_and_train_model(X_train, y_train, sequence_length, num_features, num_epochs):
    model = Sequential([
        LSTM(100, activation='relu', input_shape=(sequence_length, num_features), 
             return_sequences=True, kernel_regularizer=l2(0.01)),
        LSTM(50, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01)),
        TimeDistributed(Dense(1))
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        X_train, y_train, 
        epochs=num_epochs, 
        batch_size=32, 
        validation_split=0.2, 
        verbose=1,
        callbacks=[early_stopping]
    )
    
    return model

def make_pred_plot(csv, interval, num_epochs):
  sequence_length = 864

  # Load and preprocess the data
  data, feature_scaler, feature_columns, voltage_scaler, original_volt, original_time = load_and_preprocess_data(csv)

  print(f"Data: {len(data)}")

  # Prepare sequences for LSTM
  X, y, timestamps = prepare_sequences(data, sequence_length, feature_columns)

  # Get the correct number of features
  num_features = X.shape[2]

  # Split the data into training and testing sets
  split = int((interval - 1) * len(X) / interval)
  X_train, X_test = X[:split], X[split:]
  y_train, y_test = y[:split], y[split:]
  timestamps_train, timestamps_test = timestamps[:split], timestamps[split:]

  print(f"X: {len(X)}")

  # Create and train the model
  model = create_and_train_model(X_train, y_train, sequence_length, num_features, num_epochs)

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

  ax.plot(original_time, original_volt, label="Original Data", color='lightgray', alpha=0.5)
  ax.plot(timestamps_test.flatten(), y_test_original.flatten(), label = "Actual")
  ax.plot(timestamps_test.flatten(), predictions_original.flatten(), label = "Preds")

  ax.set_xlabel("Timestamp")
  ax.set_ylabel("Voltage")
  ax.set_title('LSTM Model: Actual vs Predicted Voltage for Next Day')
  ax.legend()
  plt.xticks(rotation=45)
  plt.tight_layout()

  return fig


# fig 1 is 86 seq on 10 epoch
# fig 2 is 864 seq on 50 epoch