# Libraries
import streamlit as st
from PIL import Image

# Confit
st.set_page_config(page_title='Project Demo', page_icon=':bar_chart:', layout='wide')

# Title
st.title('Offshore Platform Monitoring System')
st.write(
    """
    This monitoring system is a prototype tool for analyzing offshore platform sensors
    """
)

# show a picture
image = Image.open('./images/home.png')
st.image(image, caption='functions', use_column_width=True)
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