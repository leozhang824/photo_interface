import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os, re

####### IMPORTANT
### 86400 is key because it is the number of seconds in a day
### -> 1 data point / sec
### Sensor isnt very effective at this so we perform some interpolation
### LSTM requires all sequences to have equal lengths
########

def load_and_concatenate_data(folder_path, start_date, end_date):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            all_data.append(df)
    
    concatenated_df = pd.concat(all_data, ignore_index=True)
    concatenated_df = concatenated_df.sort_values('Timestamp')

    # Load the CSV file
    file_path = './ambient-data-logs-arduino/CBG25Test.csv'
    # Read the file content as plain text
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Define a list to store the parsed data
    parsed_data = []

    # Define regular expressions to match the required data
    for line in lines:
        timestamp_match = re.search(r'^\s*([\d\-:\s]+)', line)
        timestamp = timestamp_match.group(1).strip() if timestamp_match else None

        soil_moisture_match = re.search(r'^\s*[\d\-:\s]+,\s*([\d.]+)', line)
        soil_moisture = float(soil_moisture_match.group(1)) if soil_moisture_match else None

        soil_temp_match = re.search(r'Soil Temp:\s*([\d.]+)', line)
        soil_temp = float(soil_temp_match.group(1)) if soil_temp_match else None

        soil_ec_match = re.search(r'Soil Emissivity:\s*([\d.]+)', line)
        soil_ec = float(soil_ec_match.group(1)) if soil_ec_match else None

        lux_match = re.search(r'Ambient Light:\s*([\d.]+)\s*lux', line, re.IGNORECASE)
        lux_value = float(lux_match.group(1)) if lux_match else None

        ambient_temp_match = re.search(r'MCP9808 Temp:\s*([\d.]+)\s*C', line, re.IGNORECASE)
        ambient_temp = float(ambient_temp_match.group(1)) if ambient_temp_match else None

        # Append the parsed data to the list
        parsed_data.append([timestamp, soil_moisture, soil_temp, soil_ec, lux_value, ambient_temp])

    cbg_df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Soil Moisture', 'Soil Temp', 'Electrical Conductivity', 'Lux', 'Ambient Temp'])
    cbg_df['Timestamp'] = pd.to_datetime(cbg_df['Timestamp'])
    
    # Filter CBG data after the specified time
    filter_time = '2024-09-26 12:40:00'
    cbg_df = cbg_df[cbg_df['Timestamp'] > filter_time]
    cbg_df.dropna(how='all', subset=['Soil Moisture', 'Soil Temp', 'Electrical Conductivity', 'Lux', 'Ambient Temp'], inplace=True)

    # Merge the two DataFrames
    merged_df = pd.merge_asof(concatenated_df, cbg_df, on='Timestamp', direction='nearest')

    # Filter data for the specified date range
    mask = (merged_df['Timestamp'] >= start_date) & (merged_df['Timestamp'] < end_date)
    filtered_df = merged_df.loc[mask]

    return filtered_df

def resample_data(df, target_frequency='1s'):
    df = df.sort_values('Timestamp')   

    # Resample the data to the target frequency
    df_resampled = df.set_index('Timestamp').resample(target_frequency).mean().reset_index()
    
    # Fill missing values with forward fill, then backward fill
    df_resampled = df_resampled.ffill().bfill()

    return df_resampled


def plot_data(original_df, adjusted_df):
    # Determine the voltage column name
    voltage_col = 'Dev1 Channel3' if 'Dev1 Channel3' in original_df.columns else 'Voltage'

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Plot original data
    ax1.plot(original_df['Timestamp'], original_df[voltage_col], label='Original', alpha=0.7)
    ax1.set_ylabel('Voltage')
    ax1.legend()
    ax1.grid(True)
    
    # Plot adjusted data
    ax2.plot(adjusted_df['Timestamp'], adjusted_df[voltage_col], label='Adjusted to 1 point/sec', alpha=0.7, color='red')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Voltage')
    ax2.legend()
    ax2.grid(True)
    
    # Rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    
    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()

# Load and concatenate data
folder_path = 'cbg_sep25'
start_date = '2024-09-26 12:40:00'
end_date = '2024-10-03 12:40:00'

filtered_data = load_and_concatenate_data(folder_path, start_date, end_date)
resampled_data = resample_data(filtered_data)

plot_data(filtered_data, resampled_data)

# Save data
filtered_data.to_csv('ML_work/temp_data.csv', index=False)
resampled_data.to_csv('ML_work/week_data.csv', index=False)

