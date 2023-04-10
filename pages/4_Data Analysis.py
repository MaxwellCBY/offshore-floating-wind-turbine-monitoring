import streamlit as st
import pandas as pd
from cmath import nan
import matplotlib.pyplot as plt
from PIL import Image
import time
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

st.title("Time Series Classification")
st.caption("Classify time series data using machine learning algorithms")
st.header("3 DOF accelerator sensor Data")

st.header("Data Collection")

training_results = {2:0, 3:0, 6:0}
testing_results = {2:0, 3:0, 6:0}

# Load data from csv file
df = pd.read_csv("data_set_group18.csv")
df.drop(columns=['Unnamed: 0','person_id'], inplace=True)

# Load images used for plots. For source code please see Readme.md
box1 = Image.open('./images/image2.png')
box2 = Image.open('./images/image9.png')
stack1 = Image.open('./images/image4.png')
stack2 = Image.open('./images/image13.png')
dd1 = Image.open('./images/image3.png')
dd2 = Image.open('./images/image11.png') 
sc1 = Image.open('./images/pca6.png')
pca2 = Image.open('./images/pca2.png')
pca3 = Image.open('./images/pca5.png')
pca6 = Image.open('./images/pca4.png')

## Loading raw data from csv file
if st.button("Fetch Data") or 'fetch' in st.session_state.keys():
    
    st.subheader("Displaying raw wellness data from wearable")
    if 'fetch' not in st.session_state.keys():
        st.session_state['fetch'] = True
    if st.checkbox("Preview Data"):
        # Preview dataframe in the web applet
        st.dataframe(df)

    st.markdown(
        """
        > **Note:** The data is pre-collected from the wearable device and is not real-time data.
        
        **Database features**:
        - `person_id`: Unique ID for each participant
        - `exercise`: Defines type of exercise done by user
            - `0`: No exercise
            - `1`: Walking
            - `2`: Running stairs
            - `3`: Plank
            - `4`: HIIT
            - `5`: Squats
        - `heart_rate`: Heart rate collected from wearable device
        - `SPO`: Oxygen saturation collected from wearable device
        - `heart_rate_base`: Heart rate collected from commercial fingertip oximeter
        - `SPO_base`: Oxygen saturation collected from commercial fingertip oximeter
        - `X`,`Y`,`Z`: Accelerometer data collected from accompanying mobile device in acceleration (m/s^2)
        """)


    st.subheader("Data Visualization")
    st.caption("Let's see how the data looks like.")
    option = st.selectbox(
        'Choose a visulization type',
        ('none','Boxplot', 'Stacked Bar Chart', 'Density Distribution', 'Scatter Plot','PCA'))

    if option == 'Boxplot':
        st.image(box1, caption='Box Plot for spO2', use_column_width=True)
        st.image(box2, caption='Box Plot for Heart Rate', use_column_width=True)

    elif option == 'Stacked Bar Chart':
        st.image(stack1, caption='Stacked Bar Chart for spO2', use_column_width=True)
        st.image(stack2, caption='Stacked Bar Chart for Heart Rate', use_column_width=True)
    
    elif option == 'Density Distribution':
        st.image(dd1, caption='spO2 Density Distribution Plot', use_column_width=True)
        st.image(dd2, caption='Heart Rate Density Distribution Plot', use_column_width=True)

    elif option == 'Scatter Plot':
        st.image(sc1, caption='Scatter Plot for spO2, Heart Rate and XYZ', use_column_width=True)
    
    elif option == 'PCA':
        clusters = st.select_slider("Select number of clusters to visualize",[2,3,6])
    
        #if 'train' not in st.session_state.keys():
        #    st.session_state['train'] = True

        if clusters == 2:
            st.image(pca2, caption='PCA for 2 clusters', use_column_width=True)
            st.markdown(
                """
                **Cluster breakdown**:
                - `0`: No exercise, Walking, Plank, Running stairs 
                - `1`: HIIT (Mountain Climbing), Squats
                """
            )
        elif clusters == 3:
            st.image(pca3, caption='PCA for 3 clusters', use_column_width=True)
            st.markdown(
                """
                **Cluster breakdown**:
                - `0`: No exercise, Walking, Plank
                - `1`: Running stairs
                - `2`: HIIT (Mountain Climbing), Squats
                """
            )
        elif clusters == 6:
            st.image(pca6, caption='PCA for 6 clusters', use_column_width=True)
            st.markdown(
                """
                **Cluster breakdown**:
                - `0`: No exercise, Walking, Plank
                - `1`: Running stairs
                - `2`: HIIT (Mountain Climbing), Squats
                - `3`: Mountain Climbing
                - `4`: Squats
                - `5`: Mountain Climbing, Squats
                """
            )


    st.subheader("Model Training")
    st.caption("We will now train a model using the collected data for 2 clusters.")

    if st.button("Train Data") or 'train' in st.session_state.keys():
        st.markdown(
        """        
        **Model Details**:
        -  **PCA**: We use PCA to reduce the dimensionality of the data to 3D.
        -  **GMM**: We use a Gaussian Mixture Model to cluster the resulting PCA data. 

        The per-point clustering accuracy of the model is also shown below.
        """)

        def apply_pca(df_m):
            pca = PCA(n_components=3)
            principalComponents = pca.fit_transform(df_m)
            principalDf = pd.DataFrame(
                data=principalComponents,
                columns=[
                    "principal component 1",
                    "principal component 2",
                    "principal component 3",
                ],
            )
            return principalDf

        df_m = pd.read_csv("sd2_avg_bal_data_set_group18_new_1.csv")
        labels = df_m["exercise"]
        df_val = pd.read_csv("test_sd2avg_data_set_group18_1.csv")
        labels_val_n = df_val["exercise"]


        df_m = df_m.drop(
            columns=[
                "index",
                "exercise",
                "time",
                "person_id",
                "spo_base",
                "heart_rate_base",
                "absolute",
            ]
        )

        gm = GaussianMixture(
            n_components=2, covariance_type="diag", random_state=42, n_init=10
        )

        df_m = apply_pca(df_m)
        gm.fit(df_m)
        
        tr_res = ((labels==gm.predict(df_m)).sum()*100)/df_m.shape[0]

        # testing

        df_val = df_val.drop(
            columns=[
                "index",
                "exercise",
                "person_id",
                "time",
                "spo_base",
                "heart_rate_base",
                "absolute",
            ]
        )
            
        ts_res = ((labels_val_n==gm.predict(apply_pca(df_val))).sum()*100)/df_val.shape[0]
        
        st.subheader("Results")
        st.write("Training Accuracy: ", tr_res)
        st.write("Testing Accuracy: ", ts_res)

  



