import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.stats.diagnostic import kstest_normal


from outlier import downsample_func
from outlier import iqr_outliers
from outlier import plot_outliers
from outlier import zscore
from outlier import modified_zscore
from outlier import plot_zscore


def load_data():
    st.write("Upload a csv file")
    uploaded_file = st.file_uploader("Choose a file",'csv')
    use_example_file = st.checkbox("Use example file",False,help="Use in-built example file for demo")

    status = False
    if use_example_file:
        uploaded_file = "default_file.csv"
        status = True
    
    if uploaded_file:
        #st.write(uploaded_file)
        if(uploaded_file == None):
            status = False
        else:
            status = True
    to_return = [uploaded_file,status]

    return to_return

def read_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df



def main():

    global output_df
    st.title("Outlier Detection in Time-Series Data")
    status = False
    set_index  = False
    basic_done = False
    #st.write(status)
    uploaded_file,status = load_data()
    #st.write(status)
    if(status == True):
        df = read_data(uploaded_file)
        st.write(df.head())
        st.write('Choose a column to set as an index. Data of column must be in date_time ')
        col_list = list(df.columns.values)
        col_list.insert(0,'None')
        indexx = st.selectbox('Which column you want as index:',col_list)

        st.write('You Selected: ',indexx)

        try:
            df[indexx] = pd.to_datetime(df[indexx])
            set_index = True
        except:
            set_index = False
            st.write("Error select proper column")

        if(set_index == True):
            df = df.set_index(indexx)
            st.write(df.head())
            #st.write("Current frequency")
            #st.write(df.index.freq)
            try:
                st.line_chart(df,width=1000,height=500)
            except:
                st.write('Error, Choose another column')
            basic_done = True
        
        if(set_index == True):

            st.sidebar.markdown("## Downsample")
            menu = ['None','Daily','3 Days','Weekly','Fortnight','Monthly','Quaterly','Yearly']
            op = st.sidebar.selectbox('Option',menu)
            if(op != 'None'):
                menu_agg = ['None','mean','min','max','median','sum']
                op_agg = st.sidebar.selectbox('Option',menu_agg)
                if(op_agg != 'None'):
                    downsample_df = downsample_func(op,df,op_agg)
                    st.write(downsample_df.head())
                    st.line_chart(downsample_df)


            st.sidebar.markdown("## Choose dataframe")
            menu = ['Main file','Downsampled file']
            op = st.sidebar.selectbox('Option',menu)
            df_graph = df
            if(op == 'Downsampled file'):
                try:
                    df_graph = downsample_df
                except:
                    st.write('You have not used the option of downsample')
            
            st.sidebar.markdown("## Detecting outliers using visualizations")
            menu = ['None','Box plot','Boxen plot','Violin plot','Lag plot']
            op_graph = st.sidebar.selectbox('Option',menu)
            if (op_graph == 'Box plot'):
                fig = plt.figure(figsize=(10, 4))
                sns.boxplot(df_graph.iloc[:, 0],orient="h")
                st.pyplot(fig)
            elif(op_graph == 'Boxen plot'):
                #boxen plot with different depths
                for k in ['tukey',"proportion","trustworthy","full"]:
                    fig = plt.figure(figsize=(10, 4))
                    sns.boxenplot(df_graph.iloc[:, 0],k_depth=k,orient='h')
                    plt.title(k)
                    st.pyplot(fig)
            elif(op_graph == 'Violin plot'):
                fig = plt.figure(figsize=(16, 3))
                sns.violinplot(df_graph.iloc[:, 0],orient='h')
                st.pyplot(fig)
            
            elif(op_graph == 'Lag plot'):
                fig = plt.figure(figsize=(16, 3))
                lag_plot(df_graph)
                st.write(type(fig))
                st.pyplot(fig)



            st.sidebar.markdown("## Detect outlier with stats tool")
            menu = ['None','Tuckey Method','Z-score method']
            op_method = st.sidebar.selectbox('Option',menu)
            if(op_method == 'Tuckey Method'):
                st.write("The outliers as per Tuckey Method")
                outliers = iqr_outliers(df_graph)

                st.write(outliers)
                st.write(len(outliers))
                if(len(outliers) > 0):
                    plot_outlier = plot_outliers(outliers,df_graph,"Outliers using IQR with Tukey's Fences")
                    st.pyplot(plot_outlier)


            elif(op_method == 'Z-score method'):
                #Test if data is normal or not
                t_test,p_value = kstest_normal(df_graph)
                if p_value < 0.05:
                    st.write("Data is not normal")
                    outliers,transformed = modified_zscore(df_graph)
                    if(len(outliers) > 0):
                        st.write(outliers)
                        plot_outlier = plot_outliers(outliers,df_graph,"Outliers using modified Z-Score method")
                        st.pyplot(plot_outlier)

                        data = transformed['m_zscore'].values
                        plot_z_outlier = plot_zscore(data)
                        st.pyplot(plot_z_outlier)

                else:
                    st.write(" Data is normal")
                    outliers, transformed = zscore(df_graph)
                    if(len(outliers) > 0):
                        st.write(outliers)
                        plot_outlier = plot_outliers(outliers,df_graph,"Outliers using Z-Score method")
                        st.pyplot(plot_outlier)

                        data = transformed['zscore'].values
                        plot_z_outlier = plot_zscore(data)
                        st.pyplot(plot_z_outlier)

        




if __name__ == '__main__':
    main()