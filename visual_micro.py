import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from plot_ep import plot

st.set_page_config(
    page_title="Photosynthesis Data Visualizer",
    layout="wide",
)

st.title("Photosynthesis Data Visualizer")
st.write("Upload a CSV file and select desired parameters, to display seelcted graphs.")

csv_file = st.file_uploader("Choose a CSV file")

if csv_file is not None:
  # Read and ajdust csv_data
  data = pd.read_csv(csv_file)
  st.write("CSV File Name:", csv_file.name)

  # data['Timestamp'] = pd.to_datetime(data['Timestamp'])
  data.rename(columns={'Dev1 Channel3': 'Voltage'}, inplace=True)

  st.write(data)
  
  # Time & Date Selection
  col1, col2 = st.columns(2)

  # with col1:
  #   start_time = st.selectbox(label="Choose your start time", options=data["Timestamp"][::86350])
  
  # with col2:
  #   end_time = st.selectbox(label="Choose your end time", options=data["Timestamp"][::86350])
  
  # for microcontroller ver. need to actually look at data
  with col1: 
    start_date = st.date_input("Input the date that you started recording data")
    st.write(start_date)
    
  with col2:
    start_time = st.time_input("Input the time that you started recording data", value="now")
    st.write(start_time)
  
  start_datetime = datetime.datetime.combine(start_date, start_time) 
  
  # ver 2
  data["Timestamp"] = start_datetime + pd.to_timedelta(data["Time (seconds)"], unit="s")

  # Column / y-val parameter selection
  col_names = data.columns.tolist()

  col_names = [col for col in col_names if col not in ['Time (seconds)', "Timestamp"]]

  y_vals = st.multiselect(label="Choose your y-values", options=col_names)
  num_y = len(y_vals)
  y_vals_color = [0] * num_y
  
  color_cols = st.columns(num_y or 1)

  for i in range(num_y):
    with color_cols[i]:
      y_vals_color[i] = st.color_picker(label=f'Choose a color for {y_vals[i]}')


  # Plotting / VIsualizing inputted CSV data
  # if start_time  and y_vals and start_time:
  # # if start_time and start_date and y_vals:  
  #   mask = (data['Timestamp'] >= start_time) 
  filtered_data = data


  for i, col in enumerate(y_vals):
      fig, ax = plt.subplots()

      ax.plot(filtered_data['Timestamp'], filtered_data[col], color=y_vals_color[i], label=col)

      # ax.axvspan(start_time, end_time, color='grey', alpha=0.5)

      lux_threshold = 0
      lux_above = filtered_data['light'] > lux_threshold 

      in_high = False
      sub_start_time = None

      for j in range(len(filtered_data)):
          if lux_above.iloc[j] and not in_high:
              in_high = True
              sub_start_time = filtered_data['Timestamp'].iloc[j]
          elif not lux_above.iloc[j] and in_high:
              in_high = False
              sub_end_time = filtered_data['Timestamp'].iloc[j]
              ax.axvspan(sub_start_time, sub_end_time, color='yellow', alpha=0.3)

      if in_high:
          sub_end_time = filtered_data['Timestamp'].iloc[-1]
          ax.axvspan(sub_start_time, sub_end_time, color='yellow', alpha=0.3)

      ax.set_xlabel("Timestamp")
      ax.set_ylabel(f"{col}")
      ax.set_title(f"Graph Comparing Time and {col}")
      ax.legend()
      plt.xticks(rotation=45)
      plt.tight_layout()

      st.pyplot(fig)

    # # ver 1
  if st.button("Generate more interactive plot"):
    fig = plot(data, start_datetime)
    fig.show()

