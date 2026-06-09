# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 18:20:10 2023

@author: haris
"""



def PricesDK(df_prices):
    
    # Set the Sell price equal to the spot price
    df_prices["Sell"] = df_prices["SpotPriceDKK"]
    
    # Define the fixed Tax and TSO columns
    df_prices["Tax"] = 0.8
    df_prices["TSO"] = 0.1
    
    ### Add the DSO tariffs
    
    # The Low period has the same price during both summer/winter periods
    df_prices.loc[df_prices["HourDK"].dt.hour.isin([0,1,2,3,4,5]),
                  "DSO"] = 0.15
    
    # Peak period in Winter
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([1,2,3,10,11,12]))
                  & (df_prices["HourDK"].dt.hour.isin([17,18,19,20])),
                  "DSO"] = 1.35
    
    # Peak period in Summer
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9]))
                  & (df_prices["HourDK"].dt.hour.isin([17,18,19,20])),
                  "DSO"] = 0.6
    
    # High period in Winter
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([1,2,3,10,11,12]))
                  & (df_prices["HourDK"].dt.hour.isin([6,7,8,9,10,11,12,13,14,15,16,21,22,23])),
                  "DSO"] = 0.45
    
    # High period in Summer
    df_prices.loc[(df_prices["HourDK"].dt.month.isin([4,5,6,7,8,9]))
                  & (df_prices["HourDK"].dt.hour.isin([6,7,8,9,10,11,12,13,14,15,16,21,22,23])),
                  "DSO"] = 0.23
    
    # Calculate VAT
    df_prices["VAT"] = 0.25*(df_prices["Tax"]+df_prices["TSO"]+df_prices["DSO"]+df_prices["SpotPriceDKK"])
    
    # Calculate Buy price
    df_prices["Buy"] = df_prices["Tax"]+df_prices["TSO"]+df_prices["DSO"]+df_prices["SpotPriceDKK"]+df_prices["VAT"]
    
    return df_prices


def LoadData():
    
    import os
    import pandas as pd
    
    ### Load electricity prices ###
    price_path = os.path.join(os.getcwd(),'ElspotpricesEA.csv')
    df_prices = pd.read_csv(price_path)
    
    # Convert to datetime
    df_prices["HourDK"] = pd.to_datetime(df_prices["HourDK"])
    
    # Convert prices to DKK/kWh
    df_prices['SpotPriceDKK'] = df_prices['SpotPriceDKK']/1000
    
    # Filter only DK2 prices
    df_prices = df_prices.loc[df_prices['PriceArea']=="DK2"]
    
    # Keep only the local time and price columns
    df_prices = df_prices[['HourDK','SpotPriceDKK',"HourUTC"]]
    
    # Keep only 2019, 2020, 2021, 2022, 2023
    df_prices = df_prices.loc[df_prices["HourDK"].dt.year.isin([2019,2020,2021,2022,2023])]
    
    # Reset the index
    df_prices = df_prices.reset_index(drop=True)
    
    ###  Load prosumer data ###
    file_P = os.path.join(os.getcwd(),'ProsumerHourly.csv')
    df_pro = pd.read_csv(file_P)
    df_pro["TimeDK"] = pd.to_datetime(df_pro["TimeDK"])
    df_pro = df_pro.reset_index(drop=True)
    df_pro.rename(columns={'Consumption': 'Load'}, inplace=True)
    df_pro.rename(columns={'TimeDK': 'HourDK'}, inplace=True)

    return df_prices, df_pro


def Netting(df_pro, df_prices, res):
    
    import pandas as pd
    
    if res == "Yearly":
        
        # Calculate yearly price statistics
        df_prices_mean = df_prices.groupby('Year').agg({'Buy': 'mean', 'Sell': 'mean'}).reset_index()

        # Calculate yearly statistics
        df_sum = df_pro.groupby('Year').agg({'PV': 'sum', 'Load': 'sum'}).reset_index()

        # Calculate yearly Imports/Exports
        df_sum["Export"] = (df_sum["PV"] - df_sum["Load"]).where(df_sum["PV"] > df_sum["Load"], other=0)
        df_sum["Import"] = (df_sum["Load"] - df_sum["PV"]).where(df_sum["Load"] > df_sum["PV"], other=0)

        Net = pd.merge(df_prices_mean, df_sum, on='Year')
        Net['Profit'] = Net["Export"]*Net["Sell"] - Net["Import"]*Net["Buy"]
        
    elif res == "Monthly":
        
        # Calculate monthly price statistics
        df_prices_mean = df_prices.groupby(['Year','Month']).agg({'Buy': 'mean', 'Sell': 'mean'}).reset_index()

        # Calculate monthly statistics
        df_sum = df_pro.groupby(['Year','Month']).agg({'PV': 'sum', 'Load': 'sum'}).reset_index()

        # Calculate yearly Imports/Exports
        df_sum["Export"] = (df_sum["PV"] - df_sum["Load"]).where(df_sum["PV"] > df_sum["Load"], other=0)
        df_sum["Import"] = (df_sum["Load"] - df_sum["PV"]).where(df_sum["Load"] > df_sum["PV"], other=0)

        Net = pd.merge(df_prices_mean, df_sum, on=['Year','Month'])
        Net['Profit'] = Net["Export"]*Net["Sell"] - Net["Import"]*Net["Buy"]
        
        Net = Net.groupby('Year').agg({'Profit': 'sum'}).reset_index()

    elif res == "Hourly":

        # Add buy/sell prices
        df_pro["Buy"] = df_prices["Buy"]
        df_pro["Sell"] = df_prices["Sell"]

        df_pro["Export"] = (df_pro["PV"] - df_pro["Load"]).where(df_pro["PV"] > df_pro["Load"], other=0)
        df_pro["Import"] = (df_pro["Load"] - df_pro["PV"]).where(df_pro["Load"] > df_pro["PV"], other=0)
        df_pro['Profit'] = df_pro["Export"]*df_pro["Sell"] - df_pro["Import"]*df_pro["Buy"]

        Net = df_pro.groupby('Year').agg({'Profit': 'sum'}).reset_index()
        
    elif res == "No":
    
        # Add buy/sell prices
        df_pro["Buy"] = df_prices["Buy"]
        df_pro["Sell"] = df_prices["Sell"]

        df_pro['Profit'] = df_pro["PV"]*df_pro["Sell"] - df_pro["Load"]*df_pro["Buy"]

        Net = df_pro.groupby('Year').agg({'Profit': 'sum'}).reset_index()
    
    return Net


import numpy as np

def results_year(year, params):
    results = []
    p_c = 0
    p_d = 0
    X = 0
    # Iterate through each 24-hour slice of the dataframe
    for i in range(0, len(year), 24):
        # Extract the "Sell" data for the current 24-hour slice
        sell_data = year.loc[i:i+23, 'Sell'] #*****
        
        # Call the Optimizer function with the sell_data
        result, p_c, p_d, X = Optimizer(params, sell_data)
        
        # Append the result to the results array
        results.append(result)

    # Convert results to a numpy array if needed
    results = np.sum(np.array(results))
    
    return results



def Optimizer(params, p):

    import cvxpy as cp
    #convert dataframe to array
    p = p.values
    n = len(p)
    p_c = cp.Variable(n)
    p_d = cp.Variable(n)
    X   = cp.Variable(n)
    profit = cp.sum(p_d@p - p_c@p)

    constraints = [p_c >= 0, 
                   p_d >= 0, 
                   p_c <= params['Pmax'], 
                   p_d <= params['Pmax']]
    constraints += [X >= 0.1 * params['Cmax'], X <= params['Cmax']] #Here we set X to being at least 10% of Cmax (so the charge of the battery never goes under 10% of total charge)
    constraints += [X[0]==params['C_0'] + p_c[0]*params['n_c'] - p_d[0]/params['n_d']] #When the power is multiplied by the efficiency it results in the energy, this is because the time difference (delta t) is in fact one hour, so for simplicity it isn't included.
    
    constraints += [X[1:] == X[:-1] + p_c[1:]*params['n_c'] - p_d[1:]/params['n_d']]
    
    constraints += [X[n-1]>=params['C_n']]
    
    problem = cp.Problem(cp.Maximize(profit), constraints)
    problem.solve(solver=cp.ECOS)
    
    return profit.value, p_c.value, p_d.value, X.value

def ProsumerOptimizer(params, l_b, l_s, p_PV, p_L):

    import cvxpy as cp    

    n = len(l_b)
    p_c = cp.Variable(n)
    p_d = cp.Variable(n)
    p_b = cp.Variable(n)
    p_s = cp.Variable(n)
    X   = cp.Variable(n)
    cost = cp.sum(p_b@l_b - p_s@l_s)
    
    constraints = [p_c >= 0, 
                   p_d >= 0, 
                   p_c <= params['Pmax'], 
                   p_d <= params['Pmax'],
                   p_s >= 0,
                   p_b >= 0]
    constraints += [X >= 0, X <= params['Cmax']]
    constraints += [X[0]== params['C_0'] + p_c[0]*params['n_c'] - p_d[0]/params['n_d']]
    constraints += [p_PV + p_b + p_d == p_L + p_s + p_c]
    
    constraints += [X[1:] == X[:-1] + p_c[1:]*params['n_c'] - p_d[1:]/params['n_d']]
    
    constraints += [X[n-1]>=params['C_n']]
    
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.ECOS)
    
    return cost.value, p_c.value, p_d.value, p_b.value, p_s.value, X.value


