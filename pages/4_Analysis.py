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

st.title("Clsasification of DP operation status")
st.caption("Time series data analysis and classification based on 3DOF sensor data")

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
normal = Image.open('./images/normal.png')
standby = Image.open('./images/standby.png')
check = Image.open('./images/check.png')

## Loading raw data from csv file
if st.button("Fetch Data") or 'fetch' in st.session_state.keys():
    
    st.subheader("Clasify operation status of the DP")
    if 'fetch' not in st.session_state.keys():
        st.session_state['fetch'] = True
    if st.checkbox("Preview Data"):
        # Preview dataframe in the web applet
        #st.dataframe(df)
       
        # make a selection box for the user to choose which column to plot
        # the selection box has 3 options: x, y, z
        # the default option is x
        # plot the columns of x,y,z, the x axis is the time of the dataframe
        option = st.selectbox(
            'Choose a column to plot',
            ('x', 'y', 'z'))
        st.write('You selected:', option)
        fig = plt.figure()
        plt.plot(df.index,df[option])
        plt.xlabel('Time')
        plt.ylabel("Acceleration (m/s^2)")
        plt.title("Acceleration of "+option+" axis")
        st.pyplot(fig)

    st.markdown(
        """
        > 
        - `Status`: 
           - `0`: Normal
           - `1`: Stand-by
           - `2`: Check
        """)


    st.subheader("Data Visualization")
    #st.caption("Let's see how the data looks like.")
    option = st.selectbox(
        'Choose a visulization type',
        ('PCA','Normal', 'Standby', 'Check', 'Scatter Plot'))

    if option == 'Normal':
        st.image(normal, caption='DP in normal condition', use_column_width=True)

    elif option == 'Standby':
        st.image(standby, caption='DP in standby condition', use_column_width=True)
    
    elif option == 'Check':
        st.image(check, caption='DP in check condition', use_column_width=True)

    elif option == 'Scatter Plot':
        st.image(sc1, caption='Scatter Plot for XYZ', use_column_width=True)
    
    elif option == 'PCA':
        clusters = st.select_slider("Select number of clusters to visualize",[2,3,6])
    
        #if 'train' not in st.session_state.keys():
        #    st.session_state['train'] = True

        if clusters == 2:
            st.image(pca2, caption='PCA for 2 clusters', use_column_width=True)
            st.markdown(
                """
                **Cluster breakdown**:
                - `0`: Normal
                - `1`: Standby
                """
            )
        elif clusters == 3:
            st.image(pca3, caption='PCA for 3 clusters', use_column_width=True)
            st.markdown(
                """
                **Cluster breakdown**:
                - `0`: Normal
                - `1`: Standby
                - `2`: Check
                """
            )
        elif clusters == 6:
            st.image(pca6, caption='PCA for 6 clusters', use_column_width=True)
    


    st.subheader("Results")

    if st.button("Classify") or 'train' in st.session_state.keys():
        st.markdown(
        """        
        **Model Details**:
        -  **PCA**: PCA to reduce the dimensionality of the data to 3D.
        -  **GMM**: Gaussian Mixture Model to cluster the resulting PCA data. 
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
        # keep 1 decimal place
        tr_res = round(tr_res, 1)

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
        # keep 1 decimal place
        ts_res = round(ts_res, 1)
        
        # training accuracy is tr_res/tr_res %
        st.write("Score: ", round(ts_res/tr_res*100,1), "%" )
        # markdown that "Based on comparison with historical data, the DP working condition is Normal", 
        # highlight the word "Normal" in green color
        st.markdown(
            """
            The DP working condition is **Normal**.
            """
        )


