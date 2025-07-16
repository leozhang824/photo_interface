import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from train import make_pred_plot

st.set_page_config(
    page_title="Prediction Model",
    layout="wide",
)

st.title("Prediction Model")
st.write("Upload a CSV file and generate a comparison of predicted versus actual data")

csv_file = st.file_uploader("Choose a CSV file")

if csv_file is not None:
  csv = pd.read_csv(csv_file)
  st.write("CSV File Name:", csv_file.name)

  csv['Timestamp'] = pd.to_datetime(csv['Timestamp'])
 
  col1, col2, col3 = st.columns(3)

  with col1: 
    start_date = st.date_input("Input the date that you started recording data", value=csv["Timestamp"].iloc[0].date())
    
  with col2:
    start_time = st.time_input("Input the time that you started recording data", value=csv["Timestamp"].iloc[0].time())

  with col3:
    interval = st.selectbox(label="Select which day to predict", options=(5,6,7), index=2)

  col1, col2 = st.columns(2)
  with col1: 
    num_epochs = st.slider(label="Select the number of epochs (how many times the model will run)", min_value=10, max_value=60, value=50, step=5)

  start_datetime = datetime.datetime.combine(start_date, start_time) 
  end_datetime = start_datetime + datetime.timedelta(days=interval)

  mask = (csv['Timestamp'] >= start_datetime) & (csv['Timestamp'] < end_datetime)
  filtered_csv = csv.loc[mask]

  fig = make_pred_plot(filtered_csv, interval, num_epochs)

  st.pyplot(fig)


