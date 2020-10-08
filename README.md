# Time series forecast
I have implemented a ARIMA model to do the time series forecasting.

> ARIMA - its a stats model which stands for "Auto Regression Integrated Moving Average".ARIMA model uses past values to develop an equation that uses lags and lagged errors to forecast future values of the given series.

> The parameters of the ARIMA model are defined as follows:

>> p: The number of lag observations included in the model.
>> d: The number of times that the raw observations are differenced..
>> q: The size of the moving average window, also called the order of moving average.


# Instalation and running process
1. Git clone this branch to your local system.
2. Run pip install -r requirements.txt to install dependencies.
3. Run python time_series_forecast 
4. you will get the output as csv file in 'time_series_forecast/results/' folder

Also , i have already add a output file to data folder to mention the code is working fine.

# Libraries

> pandas - used for structuring the data.
> numpy -to do interpolation to fill the missing values in time series data.
> statsmodel -used for time series forecasting.

