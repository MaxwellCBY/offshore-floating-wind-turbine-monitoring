import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import io 
import requests
import scipy.stats as stats


warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = [16,5]

#the following function takes the standarized data to plot the data with threshold lines
def plot_zscore(data,d=3):
  n = len(data)
  plt.figure(figsize=(8,8))
  plt.plot(data,'k^')
  plt.plot([0,n],[d,d],'r--')
  plt.plot([0,n],[-d,-d],'r--')
  return plt


#Create a z-score function to standarize the data and filter out the extreme values based on a threshold
def zscore(df,degree=3):
   data = df.copy()
   data['zscore'] = (data -data.mean())/data.std()
   outliers = data[(data['zscore'] <= -degree ) | (data['zscore'] >= degree)]
   return outliers[data.iloc[:, 0].name],data


def modified_zscore(df,degree=3):
  data = df.copy()
  s = stats.norm.ppf(0.75)
  numerator = s*(data - data.median())
  MAD = np.abs(data - data.median()).median()
  data['m_zscore'] = numerator/MAD
  outliers = data[(data['m_zscore'] > degree ) | (data['m_zscore'] < -degree)]
  print(outliers)
  return outliers[data.iloc[:, 0].name], data



def plot_outliers(outliers,data,method='KNN',
  halignment = 'right',
  valignment = 'bottom',
  labels=False):

  ax = data.plot(alpha=0.8)
  if labels:
    for i in outliers['value'].items():
      plt.plot(i[0],i[1],'rx')
      plt.text(i[0],i[1],f'{i[0].date()}',
      horizontalalignment = halignment,
      verticalalignment = halignment)
  else:
    data.loc[outliers.index].plot(ax=ax,style='rx')

  #plt.title(f'NYC Taxi - {method}')
  #plt.xlabel('date');
  #plt.ylabel('# of passengers')
  #plt.legend(['nyc taxi','outliers'])
  plt.show()

  return plt



def iqr_outliers(data):
  q1,q3 = np.percentile(data,[25,75])
  IQR = q3-q1
  lower_fence = q1 - (1.5 * IQR)
  upper_fence = q3 + (1.5 * IQR)
  return data[(data.iloc[:, 0] > upper_fence) | (data.iloc[:, 0] <lower_fence)]


def downsample_func(option,df,op_agg='mean'):
    if(option == 'Daily'):
        if(op_agg == 'mean'):
            downsample_df = df.resample('D').mean()
        elif(op_agg == 'min'):
            downsample_df = df.resample('D').min()
        elif(op_agg == 'max'):
            downsample_df = df.resample('D').max()
        elif(op_agg == 'sum'):
            downsample_df = df.resample('D').sum()
        elif(op_agg == 'median'):
            downsample_df = df.resample('D').median()

    elif(option == '3 Days'):
        if(op_agg == 'mean'):
            downsample_df = df.resample('3D').mean()
        elif(op_agg == 'min'):
            downsample_df = df.resample('3D').min()
        elif(op_agg == 'max'):
            downsample_df = df.resample('3D').max()
        elif(op_agg == 'sum'):
            downsample_df = df.resample('3D').sum()
        elif(op_agg == 'median'):
            downsample_df = df.resample('3D').median()
    
    elif(option == 'Weekly'):
        if(op_agg == 'mean'):
            downsample_df = df.resample('W').mean()
        elif(op_agg == 'min'):
            downsample_df = df.resample('W').min()
        elif(op_agg == 'max'):
            downsample_df = df.resample('W').max()
        elif(op_agg == 'sum'):
            downsample_df = df.resample('W').sum()
        elif(op_agg == 'median'):
            downsample_df = df.resample('W').median()

    elif(option == 'Fortnight'):
        if(op_agg == 'mean'):
            downsample_df = df.resample('SM').mean()
        elif(op_agg == 'min'):
            downsample_df = df.resample('SM').min()
        elif(op_agg == 'max'):
            downsample_df = df.resample('SM').max()
        elif(op_agg == 'sum'):
            downsample_df = df.resample('SM').sum()
        elif(op_agg == 'median'):
            downsample_df = df.resample('SM').median()

    elif(option == 'Monthly'):
        if(op_agg == 'mean'):
            downsample_df = df.resample('M').mean()
        elif(op_agg == 'min'):
            downsample_df = df.resample('M').min()
        elif(op_agg == 'max'):
            downsample_df = df.resample('M').max()
        elif(op_agg == 'sum'):
            downsample_df = df.resample('M').sum()
        elif(op_agg == 'median'):
            downsample_df = df.resample('M').median()
    
    elif(option == 'Quaterly'):
        if(op_agg == 'mean'):
            downsample_df = df.resample('Q').mean()
        elif(op_agg == 'min'):
            downsample_df = df.resample('Q').min()
        elif(op_agg == 'max'):
            downsample_df = df.resample('Q').max()
        elif(op_agg == 'sum'):
            downsample_df = df.resample('Q').sum()
        elif(op_agg == 'median'):
            downsample_df = df.resample('Q').median()

    elif(option == 'Yearly'):
        if(op_agg == 'mean'):
            downsample_df = df.resample('Y').mean()
        elif(op_agg == 'min'):
            downsample_df = df.resample('Y').min()
        elif(op_agg == 'max'):
            downsample_df = df.resample('Y').max()
        elif(op_agg == 'sum'):
            downsample_df = df.resample('Y').sum()
        elif(op_agg == 'median'):
            downsample_df = df.resample('Y').median()


    return downsample_df