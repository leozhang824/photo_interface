import streamlit as st

pg = st.navigation(
  [
    st.Page("visual.py", title="Photosynthesis Data Visualizer - Regular"),
    st.Page("predict_model.py", title="Prediction Model - Regular"),
    # For microcontroller data
    st.Page("visual_micro.py", title="Photosynthesis Data Visualizer - MicroBit"),
    st.Page("predict_model_micro.py", title="Prediction Model - MicroBit"),
    st.Page("classify.py", title="Classification / Labeling Model")
  ]
)

pg.run()