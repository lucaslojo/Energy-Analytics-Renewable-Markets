# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:28:30 2024

@author: oskarjohnbruunsmith
"""


def task_1():
        
    #Importer
    from pmdarima.arima import ARIMA
    import math
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import acf
    from statsmodels.tsa.stattools import adfuller
    import pmdarima as pm
    from sklearn.metrics import mean_squared_error
    from pmdarima import pipeline, arima, model_selection
    import seaborn as sns
    import statsmodels.api as sm
    
    
    #We load and prepare the data:
    file_P = os.path.join(os.getcwd(),'Elspotprices2.csv')
    df_prices = pd.read_csv(file_P)
    df_prices["HourUTC"] = pd.to_datetime(df_prices["HourUTC"])
    df_prices = df_prices.loc[(df_prices['PriceArea']=="DK2")][["HourUTC","SpotPriceDKK"]]
    df_prices = df_prices.loc[df_prices["HourUTC"].dt.year.isin([2019,2020,2021,2022,2023])]
    df_prices = df_prices.reset_index(drop=True)
    
    file_P = os.path.join(os.getcwd(),'ProdConData.csv')
    df_data = pd.read_csv(file_P)
    df_data["HourUTC"] = pd.to_datetime(df_data["HourUTC"])
    
    df_data = df_data.loc[df_data["HourUTC"].dt.year.isin([2019,2020,2021,2022,2023])]
    df_data = df_data.reset_index(drop=True)
    
    
    
    #We define the time-periods for the training data and test data:
    start_train = pd.to_datetime('2019-01-01') 
    end_train = pd.to_datetime('2023-11-30')
    start_test = pd.to_datetime('2023-12-01')
    end_test = pd.to_datetime('2023-12-31')
    
    #We make sure the test and training data is only for the wanted time-periods:
    train_data = df_prices[(df_prices["HourUTC"] >= start_train) & (df_prices["HourUTC"] <= end_train)].reset_index(drop=True)
    test_data = df_prices[(df_prices["HourUTC"] >= start_test) & (df_prices["HourUTC"] <= end_test)].reset_index(drop=True)
    
    
    ###########################################################################
    #Task 1.1
    ###########################################################################
    
    #We define a variable train_whole that we will use for a plot later, but need to
    #define now before we make any more changes to train_data:
    train_whole = train_data
    
    # Convert index to datetime
    train_whole.index = pd.to_datetime(train_whole.index)
    
    # Filter training data for the year 2023, as we will use it later perhaps:
    train_2023 = train_whole[train_whole.index.year == 2023]
    
    
    # Calculate average spot price for each day in the training and testing periods for plot_
    train_data['Date'] = train_data['HourUTC'].dt.date
    test_data['Date'] = test_data['HourUTC'].dt.date
    
    #The arrays are then:
    train_array_plot1 = train_data.groupby('Date')['SpotPriceDKK'].mean().values
    test_array_plot1 = test_data.groupby('Date')['SpotPriceDKK'].mean().values
    
    # Visualizing the training and testing dataset with daily average spot prices:
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(train_data['HourUTC'].dt.date.unique(), train_array_plot1, label='Training set')
    plt.plot(test_data['HourUTC'].dt.date.unique(), test_array_plot1, label='Testing set', color='orange')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.xlabel('Date (yr)')
    plt.ylabel('Daily Average Spot Price (DKK)')
    plt.title('Daily Average Spot Price per Day')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
    
    #We use the following training data, as it is close in time to our testing data:
    start_train = pd.to_datetime('2023-10-01') 
    end_train = pd.to_datetime('2023-11-30')
    train_data = df_prices.loc[(df_prices['HourUTC'] >= start_train) & (df_prices['HourUTC'] <= end_train)]
    
    # Plot ACF/PACF plots
    
    #First, we set up the canvas for our two plots:
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    
    #ACF plot
    sm.graphics.tsa.plot_acf(train_data.iloc[:,1], title = "ACF", lags=40, ax=ax[0])
    
    #PACF plot
    sm.graphics.tsa.plot_pacf(train_data.iloc[:,1], title = "PACF", lags=40, ax=ax[1])
    
    plt.tight_layout()
    plt.show()
    
    #It is clear that there is a pattern in the ACF plot. We try differencing with a lag of 24 hours:
    
    # Extract spot prices from the train DataFrame:
    spot_prices = train_data
    
    # Perform seasonal differencing with a 24-hour lag:
    train24 = pm.utils.diff(spot_prices['SpotPriceDKK'], 24)
    
    #Make ACF and PACF plots for the differences data:
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    
    #ACF plot
    sm.graphics.tsa.plot_acf(train24, lags=30, title="ACF", ax=ax[0])
    
    #PACF plot
    sm.graphics.tsa.plot_pacf(train24, lags=30, title="PACF", ax=ax[1])
    
    #Plot them in the same plot:
    plt.tight_layout()
    plt.show()
    
    #We have established seasonality. We shall use auto_arima with seasonality.
    
    ###########################################################################
    #METHOD 1
    
    #We read the p and q from differenced (m=1) ACF and PACF plots.
    
    
    #Use auto_arima to find appropriate seasonal terms for our SARIMA model:
    m_S = pm.auto_arima(train_data['SpotPriceDKK'], trace = True, seasonal = True, m = 24, 
                        stepwise=True, maxiter=10)
    
    print(m_S.summary())
    
    
    #We define a function that computes the AIC given some training data, the order
    #and the seasonal order of a SARIMA model.
    
    def compute_aic(train_data, order, seasonal_order):
        # Fit the ARIMA model to your data
        model = sm.tsa.statespace.SARIMAX(train_data['SpotPriceDKK'], order=order, seasonal_order=seasonal_order)
        results = model.fit()
    
        # Extract the AIC value from the fitted model's summary
        aic = results.aic
        return aic
    
    # Define the seasonal order for the SARIMA model we have established:
    seasonal_order = (2, 0, 0, 24)
    
    #Earlier we established the value of the ARIMA model. But, we would like to
    #test the AIC values for all the possible values of P in our model.
    #OBS. if the code is taking too long to run this part of the code is a big
    #part of that.
    
    # Loop through the range of ARIMA orders
    for p in range(16, 0, -1):
        for q in range(2, 0, -1):
            order = (p, 1, q)  # Adjust the order accordingly
            aic = compute_aic(train_data, order, seasonal_order)
            print(f"AIC for ARIMA{order}x{seasonal_order}: {aic}")
    
    #By reading the AIC values, we find the lowest value (highest efficiency
    #compared to closeness) corresponds to the model SARIMA(14,1,2)(2,0,0)[24].
    
    # Manually specify the order and seasonal_order:
    m_S = ARIMA(order=(14, 1, 2), seasonal_order=(2, 0, 0, 24), trace=True, maxiter=100)
    
    # Fit the model to your training data:
    fitted_model = m_S.fit(train_data['SpotPriceDKK'])
    
    
    # Plot the diagnostics to evaluate the residuals:
    fitted_model.plot_diagnostics(lags=24)
    plt.tight_layout()
    
    
    #We define train and test to use for later:
    train = train_data["SpotPriceDKK"].values
    test = test_data["SpotPriceDKK"].values
    
    
    #The number of days in the month of december, ceiling is to ensure we
    #do not miss any values:
    N = math.ceil(len(test)/24)
    
    # Create an empty list for the 1-month ahead forecasts
    Forecasts_S = []
    
    #We create a loop that goes through all of the days of december:
        
    for i in range(N):
    
        # Generate forecast for the next time step
        frc_S   = fitted_model.predict(n_periods=24)
    
        # Append the forecast to the list
        Forecasts_S.extend(frc_S)
    
        # Update the model with new observations
        fitted_model.update(test_data['SpotPriceDKK'][i*24:(i+1)*24])
    
    

    Persistence24 = []
    train = train_data["SpotPriceDKK"].values
    test = test_data["SpotPriceDKK"].values
    
    
    #Make sure the lengths of the forecast and the peristance is as long as the test data:
    #We do this due to us using the ceiling command to determine how many times
    #to loop. Our testing data contains 721 value, and we want the forecast to
    #match it:
    
    if len(Forecasts_S) > len(test):
        Forecasts_S = Forecasts_S[:len(test)]
    


    # Initialize the persistence forecast list with the first 24 hours taken directly from the end of the training set
    if len(train) >= 24:
        Persistence24 = list(train_data['SpotPriceDKK'][-24:])  # Using the last 24 hours of training data
    
    # Now continue the forecast using test data, each point forecasting 24 hours ahead
    for i in range(len(test) - 24):
        Persistence24.append(test_data['SpotPriceDKK'].iloc[i])
    
    # Final adjustment if needed, to ensure the persistence array is not longer than the test data:
    Persistence24 = Persistence24[:len(test)]
    
    # Plotting the testing data, persistence forecast, and ARIMA forecast
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(test_data['HourUTC'], test_data['SpotPriceDKK'], label='Testing set', color='orange')
    plt.plot(test_data['HourUTC'], Persistence24, label='Persistence forecast (24h ahead)', linestyle='--', color='red')
    plt.plot(test_data['HourUTC'], Forecasts_S, label='ARIMA forecast', linestyle='--', color='blue')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.xlabel('Date')
    plt.ylabel('Spot Price (DKK)')
    plt.title('Spot Price ARIMA and persistence forecast with testing data')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
    
    #The root mean squared errors:
    RMSE_P24 = np.sqrt(mean_squared_error(Persistence24, test_data['SpotPriceDKK']))
    RMSE_F = np.sqrt(mean_squared_error(Forecasts_S, test_data['SpotPriceDKK']))
    
    #Print them:
    print("RMSE for daily persistence: ", round(RMSE_P24))
    print("RMSE for forecasts: ", round(RMSE_F))
    
    ###########################################################################
    #End of METHOD 1
    
    
    
    
    ###########################################################################
    #Method 2
    
    
    #We would like to fit the pipeline function to our training data.
    # First we preprocess with the FourierFeaturizer
    #The AutoARIMA model picks appropriate model parameters based on the
    #preprocessed data:
    pipe = pipeline.Pipeline([
        ("fourier", pm.preprocessing.FourierFeaturizer(m=24, k = 4)),
        ("arima", arima.AutoARIMA(stepwise=False, trace=1, error_action="ignore",
                                  seasonal=False, maxiter=100,
                                  suppress_warnings=True))])
    
    #We do the actual fitting:
    pipe.fit(train)
    
    
    #We have fitted our model to the data, we would now like to use it
    #to forecast values on a day-ahead basis. We do this by creating a loop
    #that loops the amount of days we have of our testing period.
    #Every loop it forecasts 24 values of spotprice into the future.
    #After the forecast has been made and saved the actual values
    #For that period is added to the model, so that it is ready
    #To make another set of 24 values of predictions:
    
    #Empty list for forecasts:
    rolling_forecast = []
    
    #N was defined earlier. The loop is:
    for i in range(N):
    
        #Generate forecast for the next time step
        forecast = pipe.predict(n_periods=24)
        
        #Append the forecast to the list
        pipe.update(test[i*24:(i+1)*24]) 
        
        #Update the model with new observations
        rolling_forecast.extend(forecast)

    
    #Make sure the lengths of the forecast is as long as the test data
    #We have to do this due to the method of calculating N earlier:
    if len(rolling_forecast) > len(test):
        rolling_forecast = rolling_forecast[:len(test)]
    
    
    # Initialize the persistence forecast list with the first 24 hours taken 
    #directly from the end of the training set
    #We use the last 24 hours of training data for the first 24 values:
    if len(train_data) >= 24:
        Persistence24 = list(train_data['SpotPriceDKK'][-24:])  
    
    # Now continue the forecast using test data, each point forecasting 24 hours ahead
    for i in range(len(test_data) - 24):
        Persistence24.append(test_data['SpotPriceDKK'].iloc[i])
    
    #Making sure that we get the right values (we have to do this due to the 
    #way N is defined):
    Persistence24 = Persistence24[:len(test)]
    

    
        
    # Plotting the testing data, persistence forecast, and SARIMA forecast
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(test_data['HourUTC'], test_data['SpotPriceDKK'], label='Testing set', color='orange')
    plt.plot(test_data['HourUTC'], Persistence24, label='Persistence forecast (24h ahead)', linestyle='--', color='red')
    plt.plot(test_data['HourUTC'], rolling_forecast, label='ARIMA forecast', linestyle='--', color='blue')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.xlabel('Date')
    plt.ylabel('Spot Price (DKK)')
    plt.title('Spot Price SARIMA (method 2) and persistence forecast with testing data')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    #The root mean squared errors:
    RMSE_P24 = np.sqrt(mean_squared_error(Persistence24, test_data['SpotPriceDKK']))
    RMSE_F = np.sqrt(mean_squared_error(rolling_forecast, test_data['SpotPriceDKK']))
    
    
    #We print them:
    print("RMSE for daily persistence: ", round(RMSE_P24))
    print("RMSE for forecasts: ", round(RMSE_F))
    
    
    
    
    ###########################################################################
    #Task 1.2
    
    
    #Firstly we want to plot the correlation matrix to find out, what 
    #exogenous variables have the highest correlation with 'SpotPriceDKK':

    #We filter the data to include only rows where 'PriceArea' is 'DK2':
    df_data_filtered = df_data[df_data['PriceArea'] == 'DK2']
    
    # Merge the two DataFrames on the 'HourUTC' column:
    df_data_with_spotprice = pd.merge(df_data_filtered, train_whole[['HourUTC', 'SpotPriceDKK']], on='HourUTC', how='left')
    
    
    # Find the index where NaN values start in the 'SpotPriceDKK' column, so
    #We don't try to find correlation where there isn't SpotPriceDKK data:
    nan_index = df_data_with_spotprice['SpotPriceDKK'].last_valid_index() + 1
    
    # Slice all columns up to the index where NaN values start:
    df_data_sliced = df_data_with_spotprice.iloc[:nan_index]
    
    # Drop the columns that aren't relevant for our DK2 price area:
    columns_to_drop = ['ExchangeNO_MWh', 'ExchangeNL_MWh', 'ExchangeGB_MWh']
    
    # Exclude non-numeric columns:
    data_numeric = df_data_sliced.select_dtypes(include=[np.number]).drop(columns_to_drop, axis=1)
    
    #Calculate the correlation matrix based on the new DataFrame containing 
    #the SpotPriceDKK column
    correlation_matrix = data_numeric.corr()
    
    # Plot the correlation matrix
    plt.figure(figsize=(14, 12))  # Adjust the figure size as needed
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix for Exogenous Variables (PriceArea DK2)')
    plt.show()
    
    # Print correlation of 'SpotPriceDKK' column with other variables:
    print(correlation_matrix['SpotPriceDKK'])
    
    
    #Extracting exogenous variable:
    exog_vars = df_data[df_data['PriceArea'] == 'DK2'][['HourUTC','OnshoreWindGe50kW_MWh']]
    
    
    # Filter exog_vars based on HourUTC column
    exog_train = exog_vars[(exog_vars['HourUTC'] >= start_train) & (exog_vars['HourUTC'] <= end_train)]
    exog_test = exog_vars[(exog_vars['HourUTC'] >= start_test) & (exog_vars['HourUTC'] <= end_test)]
    
    #List of exogenous 
    X_train = exog_train['OnshoreWindGe50kW_MWh'].values
    X_test = exog_test['OnshoreWindGe50kW_MWh'].values
    
    # Define X_train_ar as explained in the Hands on pdf
    X_train_ar = np.column_stack([np.arange(1, len(train_data)+1), X_train])
    
    pipe = pipeline.Pipeline([
        ("fourier", pm.preprocessing.FourierFeaturizer(m=24, k = 10)),
        ("arima", arima.AutoARIMA(stepwise=False, trace=1, error_action="ignore",
                                  seasonal=False, maxiter=50, 
                                  suppress_warnings=True))])
    #Fit the model to the data:
    pipe.fit(train_data['SpotPriceDKK'], X=X_train_ar)
    
    
    #Introduce empty list that we will append forecast values to:
    rolling_forecast_X = []
    
    #Loop that computes forecasts on a day-ahead basis:
        
    for i in range(N):
        #Indices that indicate where we start and stop in the data for the forecast:
        start_idx = i * 24
        end_idx = (i + 1) * 24
        #Implement a mechanisms so we don't run into errors by trying to access
        #values that aren't there:
        if end_idx > len(X_test):
            end_idx = len(X_test)
        
        #Do the slicing on the test data we want to slice (we shall use the length
        #of this to make sure we don't incur errors or make too many forecasts)
        
        X_slice = X_test[start_idx:end_idx]
    
        #Define the time periods corresponding to the sliced data
        time_range = np.arange(start_idx + 1, start_idx + 1 + len(X_slice))
        
        # Stack the time periods and the sliced data horizontally
        X_f = np.column_stack([time_range, X_slice])
        
        # Predict the forecast for the sliced data:
        forecast = pipe.predict(n_periods=len(X_slice), X=X_f)
        #Add the forecasted values to the rolling forecast list:
        rolling_forecast_X.extend(forecast)
    
        #Slice the corresponding portion of the test data:
        test_slice = test[start_idx:end_idx]
        
        # Update the pipeline model with the actual test data slice:
        pipe.update(test_slice, X=X_f)
    
    # Ensure the rolling_forecast_X has the same length as the test set
    rolling_forecast_X = rolling_forecast_X[:len(test)]
    
    
    
    # Plot the forecast, persistence forecast, and test data:
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(test_data['HourUTC'], rolling_forecast_X, color="blue", linestyle='--', label="SARIMAX Forecast")
    plt.plot(test_data['HourUTC'], test_data['SpotPriceDKK'], color="orange", label="Actual values")
    plt.plot(test_data['HourUTC'], Persistence24, color="red", label="Persistence forecast")
    plt.xlabel('Date')
    plt.ylabel('Spot Price (DKK)')
    plt.title('Spot Price forecasting with SARIMAX (method 2), 1 exogenous variable, persistence forecast and test data')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.xticks(rotation=45)
    plt.tight_layout()  
    plt.show()
    
    #Compute the RMSE value:
    RMSE_F_X = np.sqrt(mean_squared_error(rolling_forecast_X, test_data['SpotPriceDKK']))
    
    #Print both RMSE values:
    print("RMSE for daily persistence: ", RMSE_P24)
    print("RMSE for forecasts: ", RMSE_F_X)
    
    
    
    
    #We try with two exogenous variables.
    
    #We extract the exogenous variables:
    exog_vars = df_data[df_data['PriceArea'] == 'DK2'][['HourUTC','OnshoreWindGe50kW_MWh', 'GridLossDistributionMWh']]
    
    #We filter exog_vars based on the HourUTC column:
    exog_train = exog_vars[(exog_vars['HourUTC'] >= start_train) & (exog_vars['HourUTC'] <= end_train)]
    exog_test = exog_vars[(exog_vars['HourUTC'] >= start_test) & (exog_vars['HourUTC'] <= end_test)]
    
    #We extract the exogenous variables for training and testing:
    X_train_wind = exog_train['OnshoreWindGe50kW_MWh'].values
    X_test_wind = exog_test['OnshoreWindGe50kW_MWh'].values
    X_train_loss = exog_train['GridLossDistributionMWh'].values
    X_test_loss = exog_test['GridLossDistributionMWh'].values
    
    #We define X_train_ar as explained in the Hands on pdf:
    X_train_ar = np.column_stack([np.arange(1, len(train_data)+1), X_train_wind, X_train_loss])
    
    #We fit the model to the data, where we have our two exogenous variables taken
    #Into account:
        
    pipe.fit(train_data['SpotPriceDKK'], X=X_train_ar)
    
    
    #Introduce an empty rolling forecast:
    rolling_forecast_X = []
    
    #Loop that computes the forecasts for all the appropriate days on a day-ahead basis:
    for i in range(N):
        start_idx = i * 24
        end_idx = (i + 1) * 24
        if end_idx > len(X_test_wind):
            end_idx = len(X_test_wind)
        X_slice_wind = X_test_wind[start_idx:end_idx]
        X_slice_loss = X_test_loss[start_idx:end_idx]
    
        time_range = np.arange(start_idx + 1, start_idx + 1 + len(X_slice_wind))
        X_f = np.column_stack([time_range, X_slice_wind, X_slice_loss])
        forecast = pipe.predict(n_periods=len(X_slice_wind), X=X_f)
        rolling_forecast_X.extend(forecast)
    
        test_slice = test[start_idx:end_idx]
        pipe.update(test_slice, X=X_f)
    #The loop was explained for one exogenous variable, we do the same but for two
    #exogenous variables
    
    # Ensure the rolling_forecast_X has the same length as the test set
    rolling_forecast_X = rolling_forecast_X[:len(test)]
    
    # Plot the SARIMAX model with two exogenous variables:
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(test_data['HourUTC'], rolling_forecast_X, color="blue", linestyle='--', label="ARIMA Forecast")
    plt.plot(test_data['HourUTC'], test_data['SpotPriceDKK'], color="orange", label="Actual values")
    plt.plot(test_data['HourUTC'], Persistence24, color="red", label="Persistence forecast")
    plt.xlabel('Date')
    plt.ylabel('Spot Price (DKK)')
    plt.title('Spot Price forecasting with SARIMAX, 2 exogenous variables)')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()
    
    # Calculate RMSE for ARIMA forecast
    RMSE_F_X = np.sqrt(mean_squared_error(rolling_forecast_X, test_data['SpotPriceDKK']))
    print("RMSE for daily persistence: ", RMSE_P24)
    print("RMSE for ARIMA forecast: ", RMSE_F_X)
    
    
    return 


