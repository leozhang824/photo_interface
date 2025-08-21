import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


def plot(df, start_time):
    # Set the start time
    # Convert time column to actual datetime values
    df["Datetime"] = start_time + pd.to_timedelta(df["Time (seconds)"], unit="s")

    # Create an interactive plot with markers
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Datetime"], 
        y=df["voltage"], 
        mode='markers',  # Include both lines and markers
        marker=dict(size=6, color='blue'), 
        line=dict(width=2),
        name="Voltage Readings"
    ))

    # Update layout
    fig.update_layout(
        title="Microbit ADC Voltage Readings Over Time",
        xaxis_title="Time",
        yaxis_title="voltage",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        template="plotly_white"
    )

    # Function to add shaded regions
    def add_shading(fig, start_time, end_time, color):
        fig.add_shape(
            type="rect",
            x0=start_time,
            x1=end_time,
            y0=df["voltage"].min() - 0.1,
            y1=df["voltage"].max() + 0.1,
            fillcolor=color,
            opacity=0.2,
            line_width=0,
        )

    # Loop over the date range and add shading
    current_date = df['Datetime'].min().normalize()

    while current_date <= df['Datetime'].max():
        # 8 AM to 5 PM (yellow)
        add_shading(
            fig,
            current_date + timedelta(hours=8),
            current_date + timedelta(hours=20),
            "yellow"
        )
        # 5 PM to 8 AM next day (gray)
        add_shading(
            fig,
            current_date + timedelta(hours=20),
            current_date + timedelta(days=1, hours=8),
            "gray"
        )
        current_date += timedelta(days=1)


    # Show the interactive plot
    print("returning plot")
    return fig


# # Load the CSV file
# df = pd.read_csv("microbit_agorep.csv")  # Update the path if needed
# start_time = datetime(2025, 7, 9, 18, 50, 0) 

# fig = plot(df, start_time)
# fig.show