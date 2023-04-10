import streamlit as st

st.title('Sensor Dashboard')


# go to the link http://localhost:8050/ directly
# increase the size of markdown
st.markdown('<style>h1{font-size: 50px;}</style>', unsafe_allow_html=True)
st.markdown('[Wind Turbine Dashboard](http://localhost:8050/)', unsafe_allow_html=True)

# make a button for hyperlink
st.markdown('[Offshore Platform Dashboard](http://localhost:8050/)', unsafe_allow_html=True)