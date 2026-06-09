#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:19:00 2024

@author: albertfoss
"""

def task_2_2():

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.metrics import mean_squared_error
    from tensorflow.keras.callbacks import EarlyStopping

    print('Initiated the LSTMX script')

    # Load electricity price data
    file_path = os.path.join(os.getcwd(), 'Elspotprices2.csv')
    df_prices = pd.read_csv(file_path)
    df_prices["HourUTC"] = pd.to_datetime(df_prices["HourUTC"])
    df_prices = df_prices.loc[df_prices['PriceArea'] == "DK2"][["HourUTC", "SpotPriceDKK"]]


    # Load wind data
    wind_data_path = os.path.join(os.getcwd(), 'ProdConData.csv')
    df_wind = pd.read_csv(wind_data_path)
    df_wind["HourUTC"] = pd.to_datetime(df_wind["HourUTC"])

    # Filter wind data for DK2
    df_wind = df_wind.loc[df_wind['PriceArea'] == "DK2"][["HourUTC", "OnshoreWindGe50kW_MWh"]]

    # Merge datasets on HourUTC
    df = pd.merge(df_prices, df_wind, on='HourUTC', how='left')
    df['OnshoreWindGe50kW_MWh'].fillna(method='ffill', inplace=True)  # Fill missing wind data


    # Filter data by years
    df = df.loc[df["HourUTC"].dt.year.isin([2019, 2020, 2021, 2022, 2023])]
    df = df.reset_index(drop=True)
    print('Now training the LSTMX model. This can take up to 20 minutes depending on your computer CPU')
    print('Please wait...')

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['SpotPriceDKK', 'OnshoreWindGe50kW_MWh']])
    df['ScaledPrices'] = df_scaled[:, 0]

    # Split data into training and testing
    train_scaled = df_scaled[df['HourUTC'] < '2023-12-01']
    test = df[df['HourUTC'] >= '2023-12-01']
    test_scaled = df_scaled[df['HourUTC'] >= '2023-12-01']

    # Create sequences
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length), :])
            ys.append(data[i + seq_length, 0])
        return np.array(xs), np.array(ys)

    seq_length = 24  # Look back period of 24 hours
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    # LSTM model with two input features
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(seq_length, 2)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=1)
    print('Finished training the LSTMX model')

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print('Sucessfully plotted training and validation loss for the LSTMX model')

    # Predict
    predicted_test_scaled = model.predict(X_test)
    predicted_test = scaler.inverse_transform(np.column_stack((predicted_test_scaled, np.zeros_like(predicted_test_scaled))))[:, 0]

    # Persistence forecast (using raw data for clarity)
    persistence_forecast = test['SpotPriceDKK'].shift(24).bfill().values

    # Ensure the test 'HourUTC' and 'SpotPriceDKK' are properly aligned with the forecasted results
    results_df = pd.DataFrame({
        'HourUTC': test['HourUTC'][seq_length:].reset_index(drop=True),  # Reset index to align with forecast
        'ActualPriceDKK': test['SpotPriceDKK'][seq_length:].reset_index(drop=True),  # Actual Prices
        'ForecastedPriceDKK': predicted_test,
        'PersistenceForecastDKK': test['SpotPriceDKK'].shift(24).iloc[seq_length:].reset_index(drop=True)  # Persistence forecast
    })

    # Save the results to a CSV file
    results_file_path = os.path.join(os.getcwd(), 'ForecastedPrices2.csv')
    results_df.to_csv(results_file_path, index=False)
    print(f'Results saved to {results_file_path}')

    # Calculate RMSE
    rmse_lstm = np.sqrt(mean_squared_error(test['SpotPriceDKK'][seq_length:], predicted_test))
    print('The is RMSE of the LSTMX model is:')
    print(f'LSTMX RMSE: {rmse_lstm}')

    # Plotting existing results...
    plt.figure(figsize=(14, 8))
    plt.plot(test['HourUTC'], test['SpotPriceDKK'], label='Actual Prices', color='orange')
    plt.plot(test['HourUTC'][seq_length:], predicted_test, label='Predicted Prices (LSTMX)', color='green')
    plt.plot(test['HourUTC'][seq_length:], persistence_forecast[seq_length:], label='Persistence Forecast', color='purple', linestyle='--')
    plt.title('Electricity Price Forecast for December 2023 with LSTMX Model')
    plt.xlabel('DateTime')
    plt.ylabel('Price (DKK/MWh)')
    plt.legend()
    plt.show()
    print('Sucessfully plotted the LSTMX model forecast')
    print('Finished running the LSTMX script')

    
    return

