import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import numpy as np
from train_micro import make_pred_plot

def output(csv_file):
  min_epoch = 20
  max_epoch = 110
  min_lr = 0.001
  max_lr = 0.006

  for num_epochs in range(min_epoch, max_epoch, 10):
    for learning_rate in np.arange(min_lr, max_lr, 0.001):
      if num_epochs+learning_rate in st.session_state:
        del st.session_state[num_epochs+learning_rate]
    
  csv = pd.read_csv(csv_file)
  
  st.write("CSV File Name:", csv_file.name)

  figures = make_pred_plot(start_datetime, filter_datetime, csv, min_epoch, max_epoch, min_lr, max_lr)
  
  for num_epochs in range(min_epoch, max_epoch, 10):
    for learning_rate in np.arange(min_lr, max_lr, 0.001):
      st.session_state[num_epochs+learning_rate] = figures[num_epochs+learning_rate]

  st.session_state["uploaded"] = csv_file.name

st.set_page_config(
    page_title="Prediction Model",
    layout="wide",
)

st.title("Prediction Model")
st.write("Upload a CSV file and generate a comparison of predicted versus actual data")

csv_file = st.file_uploader("Choose a CSV file")


col1, col2, col3 = st.columns(3)

with col1: 
  start_date = st.date_input("Input the date that you started recording data")
  
with col2:
  start_time = st.time_input("Input the time that you started recording data")

col1, col2 = st.columns(2)

with col1: 
  filter_date = st.date_input("Input the date that you started using data")
  
with col2:
  filter_time = st.time_input("Input the time that you started using data")

start_datetime = datetime.datetime.combine(start_date, start_time) 

filter_datetime = datetime.datetime.combine(filter_date, filter_time) 

if st.button("Rerun"):
    if "uploaded" in st.session_state:
      del st.session_state["uploaded"]
    if csv_file is not None:
      output(csv_file)

if csv_file is not None:
  if "uploaded" not in st.session_state or st.session_state["uploaded"] != csv_file.name:
      output(csv_file)

  st.write("All Done!")

col1, col2 = st.columns(2)
with col1: 
  num_epochs = st.slider(label="Select the number of epochs (how many times the model will run)", min_value=20, max_value=100, value=50, step=10)
with col2: 
  learning_rate = st.slider(label="Select the learning rate (how fast the model will improve)", min_value=0.001, max_value=0.005, value=0.003, step=0.001, format="%.3f")


if (num_epochs + learning_rate) not in st.session_state:
  st.write("Not ready yet")
else:
  st.pyplot(st.session_state[num_epochs + learning_rate])
