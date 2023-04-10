# Libraries
import streamlit as st
from PIL import Image

# Confit
st.set_page_config(page_title='Project Demo', page_icon=':bar_chart:', layout='wide')

# Title
st.title('Automated offshore monitoring system')


st.write(
    """
    This monitoring system is a prototype tool for analyzing offshore sensors, and the function includes,
    """
)

st.subheader('GPS Positioning')
st.write(
    """
    Visualize the GPS position of the assets.
    """
)

st.subheader('Sensor Dashboard')
st.write(
    """
    Show relevant sensor data from selected asset.
    """
)
st.subheader('Data Processing')
st.write(
    """
    Resample, denoise and detect outliers.
    """
)

st.subheader('Data Analysis')
st.write(
    """
    Machine learning Algorithms based on selected sensor.
    Mainly time series classification based on selected features.
    """
)

st.subheader('Auto Monitoring')
st.write(
    """
    Predict the sensor data based on data analysis and visualization the detion result.
    """
)   
st.subheader('Contact')
c1, c2, = st.columns(2)
with c1:
    st.info('**Name: Boyu Cui**', icon="📊")
with c2:
    st.info('**Phone: +47 96704305**', icon="📈")
c1,c2, = st.columns(2)
with c1:
    st.info('**Address: University of Stavanger**', icon="📈")
with c2:
    st.info('**Email: boyu.cui@uis.no**', icon="📊")