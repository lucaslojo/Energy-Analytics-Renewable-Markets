#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:19:00 2024

@author: albertfoss
"""

def task_2_1():


    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.metrics import mean_squared_error
    from tensorflow.keras.callbacks import EarlyStopping
    
    print('Initiated the LSTM script')
    
    
    # Load and preprocess data
    file_path = os.path.join(os.getcwd(),'Elspotprices2.csv')
    df_prices = pd.read_csv(file_path)
    df_prices["HourUTC"] = pd.to_datetime(df_prices["HourUTC"])
    df_prices = df_prices.loc[df_prices['PriceArea'] == "DK2"][["HourUTC", "SpotPriceDKK"]]
    df_prices = df_prices.loc[df_prices["HourUTC"].dt.year.isin([2019, 2020, 2021, 2022, 2023])]
    df_prices = df_prices.reset_index(drop=True)
    
    print('Now training the LSTM model. This can take up to 20 minutes depending on your computer CPU')
    print('Please wait...')
    
    # Separate the data into training and testing datasets
    train_prices = df_prices[df_prices['HourUTC'] < '2023-12-01']
    test_prices = df_prices[df_prices['HourUTC'] >= '2023-12-01']
    
    # Prepare the scaler using only the training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_prices['SpotPriceDKK'].values.reshape(-1, 1))
    
    # Persistence Model Forecast for December
    persistence_forecast = test_prices['SpotPriceDKK'].shift(24).bfill().values  # Shift by 24 hours for day-ahead forecast
    
    # Function to create sequences
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length)])
            ys.append(data[i + seq_length])
        return np.array(xs), np.array(ys)
    
    seq_length = 24  # Use last 24 hours to predict the next hour
    
    # Create training sequences
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_train = X_train.reshape((X_train.shape[0], seq_length, 1))
    
    # LSTM model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print('Finished training the LSTM model')
    
    
    # Plot: Training and Validation Loss vs. Epochs
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print('Sucessfully plotted training and validation loss for the LSTM model')
    
    
    # Prepare test data for prediction
    test_data = test_prices['SpotPriceDKK'].values
    X_test, y_test = create_sequences(test_data, seq_length)
    X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(-1, seq_length, 1)
    
    # Make predictions
    predicted_test_scaled = model.predict(X_test)
    predicted_test = scaler.inverse_transform(predicted_test_scaled).flatten()
    
    # Calculate RMSE for LSTM and persistence model
    rmse_lstm = np.sqrt(mean_squared_error(y_test, predicted_test))
    rmse_persistence = np.sqrt(mean_squared_error(test_prices['SpotPriceDKK'][seq_length:], persistence_forecast[seq_length:]))
    print('The is RMSE of the LSTM model is:')
    print(f'LSTM RMSE: {rmse_lstm}')
    
    # Plot 1: December prices for the actual prices, LSTM model forecast, and persistence model forecast
    plt.figure(figsize=(14, 8))
    plt.plot(test_prices['HourUTC'], test_prices['SpotPriceDKK'], label='Real Hourly Price', color='orange')
    plt.plot(test_prices['HourUTC'][seq_length:], predicted_test, label='Predicted Hourly Price (LSTM)', color='green')
    plt.plot(test_prices['HourUTC'][seq_length:], persistence_forecast[seq_length:], label='Persistence Forecast', color='purple', linestyle='--')
    plt.title('Electricity Price Forecast for December 2023')
    plt.xlabel('DateTime')
    plt.ylabel('Price (DKK/MWh)')
    plt.legend()
    plt.show()
    print('Sucessfully plotted the LSTM model forecast')
    print('Finished running the LSTM script')
    
    
    return
    
    
