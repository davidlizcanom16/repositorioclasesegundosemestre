#Example 2
import streamlit as st
import plotly.express as px
import pandas as pd

# Load Sample Data
df = px.data.gapminder()

# Title
st.title("Interactive Dashboard with Streamlit & Plotly")

# Select Year with Slider
year = st.slider("Select Year:", int(df["year"].min()), int(df["year"].max()), int(df["year"].min()))

# Filter Data
filtered_df = df[df.year == year]

# Create Plotly Scatter Plot
fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                 hover_name="country", log_x=True, size_max=60)

# Display Plot
st.plotly_chart(fig)

