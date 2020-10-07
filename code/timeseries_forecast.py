import pandas as pd
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import configparser




def get_data(filenames,sensor):
    """
    Function
    --------
    reading the data from csv's and strucrure and interpolate the data.
    Parameters
    -------
    filenames:list of the input filenames
    sensor is like hum,temp etc

    Returns
    -------
    list of the dataframes
    """
    
    list_of_df =[]
    for file in filenames:
        df= pd.read_csv('data/'+file,parse_dates=[2])
        df_updated =filter_and_structure_the_data (df,sensor)
        list_of_df.append(df_updated)
        
    return list_of_df


def filter_and_structure_the_data (df,sensor):
    

    df['datetime'] = df['datetime'].apply(lambda x:datetime.strptime(x.strftime('%Y-%m-%d %H'),'%Y-%m-%d %H'))
    df= df[df['sensor'] == sensor]
    df_updated =pd.pivot_table(df, values = 'value', index=['datetime'], columns = 'sensor').reset_index()
    df_updated =df_updated.set_index('datetime')
    
    df_updated=df_updated.resample('H').interpolate(method='linear')
    
    return df_updated
        

def get_train_ready_data(source1_df,source2_df,iot_data_df ):
    """
    Function
    --------
    creating the model input ready data from the different source of sensors.
    Parameters
    -------
    source1_df,source2_df,iot_data_df : different source of sensor data.
    Returns
    -------
    
    """


    training_data, test_data = iot_data_df[:int(len(source1_df)*0.8)],iot_data_df[int(len(source1_df)*0.8):]
    
    test_data=test_data.reset_index()
    test_data['index']=test_data.index
    
    exogenous_source1_train,exogenous_source1_forecast = source1_df[:int(len(source1_df)*0.8)],source1_df[int(len(source1_df)*0.8):]
    exogenous_source2_train,exogenous_source2_forecast = source2_df[:int(len(source1_df)*0.8)],source2_df[int(len(source1_df)*0.8):]
    
    exogenous_train = np.concatenate([exogenous_source1_train.values,exogenous_source2_train.values],axis = 1)
    exogenous_forecast = np.concatenate([exogenous_source1_forecast.values,exogenous_source2_forecast.values],axis = 1)
    
    return training_data,test_data,exogenous_train ,exogenous_forecast 
   
    
    
def ARIMA_model(training_data,exogenous_value,order):
    """
    Function
    --------
    create a arima model using training data and exogenous values.

    Parameters
    -------
    training_data
    exogenous_value
    order : its model parameter to decide the value for (p,d,q)
    Returns
    -------

    """

    arima_model = sm.tsa.statespace.SARIMAX(endog=training_data.values, exog=exogenous_value, order=order).fit()
    
    return arima_model


def get_forecast_data(x,exogenous_forecast,training_data,exogenous_train):
    
    
    """
    Function
    --------
    create a arima model and get forecast data using exogeneous values.
    
    Parameters
    -------
    x: indexing  for exogeneous values.
    exogenous_forecast : list of exogenous forecast values.
    
    Returns
    -------
    forecast value
    """

    arima_model = ARIMA_model(training_data,exogenous_train,(1,0,0))
    forecast_data = arima_model.forecast( exog=exogenous_forecast[x])[0]
    return forecast_data
    
def model_evaluation(test_data):
    
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    root_mean_squared_error = sqrt(mean_squared_error(test_data.iloc[:,-2].values,test_data.iloc[:,-1].values))
    
    mean_absolute_percentage_error = np.mean(np.abs((test_data.iloc[:,-2].values - test_data.iloc[:,-1].values)) / test_data.iloc[:,-2].values) * 100
    
    return root_mean_squared_error,mean_absolute_percentage_error


def main():
    
    
    config = configparser.ConfigParser()
    config.read('config.ini')

    print(config)
    filenames = config['Input']['filenames'].split(',')
    sensor = config['Input']['sensor']
    
    source1_df,source2_df,iot_data_df =get_data(filenames,sensor)
    training_data,test_data,exogenous_train ,exogenous_forecast   =get_train_ready_data(source1_df,source2_df,iot_data_df )
    test_data=test_data.iloc[:10]
    test_data[sensor+'_forecast']=test_data['index'][:10].apply(lambda x: get_forecast_data(x,exogenous_forecast,training_data,exogenous_train))
    test_data=test_data.drop('index',axis=1)
    
    root_mean_squared_error,mean_absolute_percentage_error= model_evaluation(test_data)
    print('root_mean_squared_error:',root_mean_squared_error,'root_mean_squared_error:',mean_absolute_percentage_error)
    test_data.to_csv('results/forecast_data.csv')
