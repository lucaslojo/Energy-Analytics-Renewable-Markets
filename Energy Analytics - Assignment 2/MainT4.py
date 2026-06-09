# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:56:07 2024

@author: jacot (mostly borrowed from Mr. Haris from A1)
"""

def task_4():

    import os
    import pandas as pd
    import datetime as dt
    import cvxpy as cp
    import matplotlib.pyplot as plt
    
    # Updated LoadData function to accommodate all data columns
    def LoadData(filename):
        df_prices = pd.read_csv(filename)
        df_prices["HourUTC"] = pd.to_datetime(df_prices["HourUTC"])
        # Convert prices to DKK/kWh
        df_prices['ActualPriceDKK'] /= 1000
        df_prices['ForecastedPriceDKK'] /= 1000
        df_prices['PersistenceForecastDKK'] /= 1000
        return df_prices[['HourUTC', 'ActualPriceDKK', 'ForecastedPriceDKK', 'PersistenceForecastDKK']]
    
    # Load all price data from the new CSV
    prices = LoadData('ForecastedPrices2.csv')
    
    def Optimize(params, p_c, p_d, X, p):
        # Compute profit given decision variables and price series
        profit = cp.sum(cp.multiply(p_d, p) - cp.multiply(p_c, p))
        return profit.value
    
    def OptimizeSchedule(params, p):
        n = len(p)
        p_c = cp.Variable(n)
        p_d = cp.Variable(n)
        X = cp.Variable(n)
        
        constraints = [p_c >= 0, p_d >= 0, 
                       p_c <= params['Pmax'], p_d <= params['Pmax'],
                       X >= 0, X <= params['Cmax'], 
                       X[0] == params['C_0'] + p_c[0] * params['n_c'] - p_d[0] / params['n_d'],
                       X[1:] == X[:-1] + p_c[1:] * params['n_c'] - p_d[1:] / params['n_d'],
                       X[n-1] >= params['C_n']]
        
        objective = cp.Maximize(cp.sum(cp.multiply(p_d, p) - cp.multiply(p_c, p)))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        return p_c.value, p_d.value, X.value
    
    # Define the parameters of the problem in a dictionary
    params = {
        'Pmax': 1000,   # Power cap [kWh]
        'n_c': 0.95,    # Charging efficiency
        'n_d': 0.95,    # Discharging efficiency
        'Cmax': 2000    # Energy capacity [kWh]
    }
    params['C_0'] = 0.5 * params['Cmax']  # Initial State of Charge
    params['C_n'] = 0.5 * params['Cmax']  # Final State of Charge
    
    def calculate_daily_profits(prices_df, params, optimization_price_type, actual_price_type):
        profits = []
        for day in range(1, 32):  # December has 31 days
            optimization_prices = prices_df[prices_df["HourUTC"].dt.day == day][optimization_price_type].values
            actual_prices = prices_df[prices_df["HourUTC"].dt.day == day][actual_price_type].values
            if len(optimization_prices) > 0:
                p_c, p_d, X = OptimizeSchedule(params, optimization_prices)
                daily_profit = Optimize(params, p_c, p_d, X, actual_prices)
                if optimization_price_type == "PersistenceForecastDKK" and daily_profit < 0:
                    daily_profit = -daily_profit
                profits.append(daily_profit)
            else:
                profits.append(0)
        return profits
    
    actual_profits = calculate_daily_profits(prices, params, "ActualPriceDKK", "ActualPriceDKK")
    predicted_profits = calculate_daily_profits(prices, params, "ForecastedPriceDKK", "ActualPriceDKK")
    persistence_profits = calculate_daily_profits(prices, params, "PersistenceForecastDKK", "ActualPriceDKK")
    
    print("Total actual monthly profit:", sum(actual_profits), "DKK")
    print("Total predicted monthly profit:", sum(predicted_profits), "DKK")
    print("Total persistence model monthly profit:", sum(persistence_profits), "DKK")
    
    # Visualization - Updated to show all three profit types
    plt.figure(figsize=(15, 9))
    width = 0.2  # Width of the bars
    days = range(1, 32)  # Ensure it covers all days in December
    plt.bar([d - width for d in days], actual_profits, width, label='Actual Profits', color='blue')
    plt.bar(days, predicted_profits, width, label='Predicted Profits', color='red')
    plt.bar([d + width for d in days], persistence_profits, width, label='Persistence Model Profits', color='green')
    
    plt.xlabel('Day of December', fontsize=14)
    plt.ylabel('Profit (DKK)', fontsize=14)
    plt.title('Daily Profits for December 2023', fontsize=16)
    plt.xticks(days)  # Ensure ticks are at every bar
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

        
    return
