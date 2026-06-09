#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:31:39 2024

@author: lucaslojoiglesias
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:06:49 2024

@author: oskarjohnbruunsmith
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway
import datetime as dt
from UsefulFunctions import LoadData
from UsefulFunctions import PricesDK
from UsefulFunctions import Netting
from UsefulFunctions import Optimizer
from UsefulFunctions import ProsumerOptimizer
#from UsefulFunctions import compute_profit_for_year
from UsefulFunctions import results_year
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from sklearn.linear_model import LinearRegression




#######################
#######Task 1##########
#######################


#1.1 Use the spot price data from years 2019, 2020, 2021, 2022 and 2023. 
#The data is contained in the ElspotpricesEA csv file. Find the average price 
#for each year in area DK2. Present your results in a figure, expressing 
#prices in DKK/MWh, and discuss your findings (evolution of prices over the years).


#Importing data into a dataframe from ElspotpricesEA.csv
df = pd.read_csv('ElspotpricesEA.csv')

#We have to change the column into datetime format, so we can easily filter by
#year

df['HourUTC'] = pd.to_datetime(df['HourUTC'])


df = df[df['HourUTC'].dt.year != 2018] #We want to filter the data from 2018 away.
#Deleting one value wil not make a significant difference in our data analysis

df = df[df['PriceArea'] == 'DK2'] #Filter for DK2

df_2019 = df[df['HourUTC'].dt.year == 2019] #Split the dataframe into the respective years
df_2020 = df[df['HourUTC'].dt.year == 2020]
df_2021 = df[df['HourUTC'].dt.year == 2021]
df_2022 = df[df['HourUTC'].dt.year == 2022]
df_2023 = df[df['HourUTC'].dt.year == 2023]

avg_price_2019 = np.mean(df_2019['SpotPriceDKK']) #Compute avg price for respective years
avg_price_2020 = np.mean(df_2020['SpotPriceDKK'])
avg_price_2021 = np.mean(df_2021['SpotPriceDKK'])
avg_price_2022 = np.mean(df_2022['SpotPriceDKK'])
avg_price_2023 = np.mean(df_2023['SpotPriceDKK'])
              

#Plot them as a function of year
avg_prices = np.array([avg_price_2019, avg_price_2020, avg_price_2021, avg_price_2022, avg_price_2023])
year = np.array([2019,2020,2021,2022,2023])
plt.bar(year, avg_prices, label='Average spotprices in DK2')
plt.xlabel('Year (yr)')
plt.ylabel('Average spotprices in DK2 (DKK)')
plt.title('Average spotprices in DK2 as a function of year (DKK/yr)')
plt.grid(True)
plt.xticks(year)
plt.show()


#We can use statistics to find out if this difference in mean is statistically significant.
#We pick one-way ANOVA.We call each year a group, and the independent variable is the year.

#Perform the one-way anova
f_statistic, p_value = f_oneway(df_2019['SpotPriceDKK'], df_2020['SpotPriceDKK'], df_2021['SpotPriceDKK'], df_2022['SpotPriceDKK'], df_2023['SpotPriceDKK'])

# Interpret the results
alpha = 0.05
print("p-value: {:.20f}".format(p_value))

if p_value < alpha:
    print("Reject the null hypothesis. There are statistically significant differences in average spot prices among the different years.")
else:
    print("Fail to reject the null hypothesis. There are no statistically significant differences in average spot prices among the different years.")
#The p-value is printed as '0' no matter the number of decimal places I set
#This indicates that the actual p-value is extremely close to zero. It is likely
#smaller than the precision that Python can represent.




#1.2 Calculate and plot the average spot price per hour of day for each of the 
#5 considered years (24x5 values) in DK2; discuss the results. Which year 
#seems more attractive for using a battery? Why?


# 1. Extract year from the 'HourUTC' column
df['Year'] = df['HourUTC'].dt.year

# 2. Initialize an empty dictionary to store hourly average prices for each year
hourly_avg_prices_by_year = {}

# 3. Loop through each year
for year in df['Year'].unique():
    # Filter data for the current year
    year_data = df[df['Year'] == year]
    
    # Group data by hour of the day and calculate the average spot price
    hourly_avg_prices = year_data.groupby(year_data['HourUTC'].dt.hour)['SpotPriceDKK'].mean()
    
    # Store hourly average prices for the current year in the dictionary
    hourly_avg_prices_by_year[year] = hourly_avg_prices

# 4. Plot the average spot price per hour of the day for each year
plt.figure(figsize=(10, 6))
for year, hourly_avg_prices in hourly_avg_prices_by_year.items():
    plt.plot(hourly_avg_prices.index, hourly_avg_prices.values, label=str(year))

plt.xlabel('Hour of the Day (h)')
plt.ylabel('Average Spot Price (DKK)')
plt.title('Average Spot Price per Hour of Day for Each Year in DK2 (DKK/h)')
plt.xticks(range(24))  # Set x-ticks to show each hour of the day
plt.grid(True)
plt.legend(title='Year')
plt.show()

#Computing the standard deviations and printing them:

std_devs = []  # Initialize empty lists to store the standard deviations

# Loop through each year in the dictionary of hourly average prices
for year, hourly_avg_prices in hourly_avg_prices_by_year.items():
    # Calculate the standard deviation of hourly average prices for the current year
    std_dev = np.std(hourly_avg_prices.values)
    
    # Append the standard deviation to the list
    std_devs.append(std_dev)

# Print the standard deviations
for i, year in enumerate(hourly_avg_prices_by_year.keys()):
    print(f"Year {year}: Standard Deviation = {std_devs[i]:.2f} DKK")


#######################
#######Task 2##########
#######################


#2.1 Optimize the operation of your battery for each day of the five years 
#(2019, 2020, 2021, 2022 and 2023). Present your results on an aggregated 
#yearly basis. Which year is more profitable? Can you provide a possible 
#explanation?


#To solve the problem we have changed the Optimizer function and have added the following constraint:
#constraints += [X >= 0.1 * params['Cmax'], X <= params['Cmax']]
#Where we set X to being at least 10% of Cmax (so the charge of the battery never goes under 10% of total charge, and the maximum is 100% of the total charge)


    
df_prices, df_pro = LoadData() #importing spotprice and sell data
df_prices = PricesDK(df_prices) # Adjust prices with tariffs
df_prices = df_prices[["HourDK","Sell","Buy"]] # Keep only buy/sell prices


# Define the parameters of the problem in a dictionary
params = {
    'Pmax': 5,
    'n_c': 0.99,
    'n_d': 0.99,
    'Cmax': 10
}
params['C_0'] = 0.5 * params['Cmax']
params['C_n'] = 0.5 * params['Cmax']


# Define year-wise DataFrames for df_prices, where we reset the index to make our loops work.
df_2019 = df_prices[df_prices['HourDK'].dt.year == 2019].reset_index(drop=True)
df_2020 = df_prices[df_prices['HourDK'].dt.year == 2020].reset_index(drop=True)
df_2021 = df_prices[df_prices['HourDK'].dt.year == 2021].reset_index(drop=True)
df_2022 = df_prices[df_prices['HourDK'].dt.year == 2022].reset_index(drop=True)
df_2023 = df_prices[df_prices['HourDK'].dt.year == 2023].reset_index(drop=True)

dataframes = [df_2019, df_2020, df_2021, df_2022, df_2023] # Define a list of dataframes for the five years

results_per_year = [] # Initialize an empty array to store aggregated results for each year

# Iterate through each DataFrame in the list
for year in dataframes:
    # Initialize an empty list to store results for the current year
    results = []
    
    # Iterate through 24-hour slices for the current year
    for i in range(0, len(year['Sell']), 24): #This means we don't make sure that the summertime/winter 
    # time shift is taken into account on the specific days where it is relevant, but the income 
    #from that hour will still be taken into account, as it will be included 
    #in another day. This will not have a significant impact on our results 
    #from the data analysis.
        # Extract the "Sell" data for the current 24-hour slice
        sell_data = year.loc[i:i+23, 'Sell']
        
        # Call the Optimizer function with the sell_data
        result, p_c, p_d, X = Optimizer(params, sell_data)
        
        # Append the result to the results list for the current year
        results.append(result)
    
    # Sum up the results for the current year and append to the results_per_year array
    results_per_year.append(np.sum(np.array(results)))

# Convert the results_per_year list to a numpy array
results_array = np.array(results_per_year)


#We now want to plot it:

# Plot the results as a bar chart with gridlines
plt.bar(np.arange(2019, 2024), results_array, color='skyblue', edgecolor='black')
plt.xlabel('Year (yr)')
plt.ylabel('Aggregated Profit (DKK)')
plt.title('Aggregated Profit for Each Year (DKK/yr)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(2019, 2024))
plt.tight_layout()
plt.show()



#2.2 Do you see any correlation between the profits of each day and the 
#average electricity price of that day? Discuss the result and what 
#conclusion you can make. You may also test whether there is stronger 
#correlation with other metrics which are based on prices. Tip: you can 
#use a scatterplot to visualize correlation.


#Now we would like to make an array of the profits for every day of 
#the year, and then we wouldl like to compute the average electricity price
#for that day. We shall plot the profit as a function of the average electricity price


#First, we will compute the profits for all days in our data, simply by defining
#the year as the data for all years:
year = df_prices

# Convert 'HourDK' column to datetime object
df_prices['HourDK'] = pd.to_datetime(df_prices['HourDK'])

# Extract date from the 'HourDK' column
df_prices['Date'] = df_prices['HourDK'].dt.date

# Initialize an empty list to store results for each day
daily_results = []

# Iterate through 24-hour slices for each day
for i in range(0, len(df_prices), 24):
    # Extract the "Sell" data for the current 24-hour slice
    sell_data = df_prices.loc[i:i+23, 'Sell']
    
    # Call the Optimizer function with the sell_data
    result, _, _, _ = Optimizer(params, sell_data)
    
    # Append the result to the daily_results list
    daily_results.append(result)

# Calculate the average electricity price for each day
daily_avg_prices = df_prices.groupby('Date')['Sell'].mean()

# Plot the profits for the day as a function of average electricity price for that day
plt.figure(figsize=(10, 6))
plt.scatter(daily_avg_prices, daily_results, color='orange', edgecolor='black')
plt.xlabel('Average Electricity Price for Given Day (DKK)')
plt.ylabel('Profits (DKK)')
plt.title('Profits as a Function of Average Electricity Price (-)')
plt.grid(True)
plt.show()

#We can compute the correlation:
# Calculate the correlation coefficient
correlation = np.corrcoef(daily_avg_prices, daily_results)[0, 1]

print("Correlation coefficient:", correlation) #0.7906272472064145



#Let us try with standard deviation for everyday instead:
# Calculate the standard deviation for each day
daily_std_dev = df_prices.groupby('Date')['Sell'].std()

# Plot the profits for the day as a function of average electricity price for that day
plt.figure(figsize=(10, 6))
plt.scatter(daily_std_dev, daily_results, color='orange', edgecolor='black')
plt.xlabel('Standard deviation of price for given day (DKK)')
plt.ylabel('Profits (DKK)')
plt.title('Profits as a Function of standard deviation (-)')
plt.grid(True)
plt.show()

#We can compute the correlation:
# Calculate the correlation coefficient
correlation = np.corrcoef(daily_std_dev, daily_results)[0, 1]

print("Correlation coefficient:", correlation) #0.9282853026574253



#2.3 Re-calculate your results with ηc = ηd = 0.95 and ηc = ηd = 0.90. 
##Present the aggregated profit for each year and for each of the three 
#efficiency cases in a plot (15 values to show). Discuss your findings and 
#explain the effect of efficiency on profits. Does efficiency or the overall 
#price level have a larger impact on profits?


# Define the parameters of the problem in a dictionary
params_095 = {
    'Pmax': 5,
    'n_c': 0.95,
    'n_d': 0.95,
    'Cmax': 10
}
params_095['C_0'] = 0.5 * params_095['Cmax']
params_095['C_n'] = 0.5 * params_095['Cmax']

params_090 = {
    'Pmax': 5,
    'n_c': 0.90,
    'n_d': 0.90,
    'Cmax': 10
}
params_090['C_0'] = 0.5 * params_090['Cmax']
params_090['C_n'] = 0.5 * params_090['Cmax']

# Define year-wise DataFrames for df_prices
dataframes = [df_2019, df_2020, df_2021, df_2022, df_2023]

# Initialize empty arrays to store agregated results for each year and efficiency case
results_per_year_095 = []
results_per_year_090 = []

# Iterate through each DataFrame in the list
for year_data in dataframes:
    # Initialize empty lists to store results for the current year and efficiency case
    results_095 = []
    results_090 = []
    
    # Iterate through 24-hour slices for the current yeer
    for i in range(0, len(year_data['Sell']), 24):
        # Extract the "Sell" data for the current 24-hour slice
        sell_data = year_data.loc[i:i+23, 'Sell']
        
        # Call the Optimizer function with the sell_data and different efficiency parameters
        result_095, _, _, _ = Optimizer(params_095, sell_data)
        result_090, _, _, _ = Optimizer(params_090, sell_data)
        
        # Append the results to the lists for the current year and efficiency case
        results_095.append(result_095)
        results_090.append(result_090)
    
    # Sum up the results for the current year and append to the results_per_year arrays
    results_per_year_095.append(np.sum(np.array(results_095)))
    results_per_year_090.append(np.sum(np.array(results_090)))

# Converting the results_per_year lists to numpy arays
results_array_095 = np.array(results_per_year_095)
results_array_090 = np.array(results_per_year_090)

# Plot the results as bar charts with gridlines
plt.figure(figsize=(10, 6))
plt.bar(np.arange(2019, 2024) - 0.2, results_array, color='skyblue', edgecolor='black', width=0.4, label='ηc = ηd = 0.99')
plt.bar(np.arange(2019, 2024) + 0.2, results_array_095, color='salmon', edgecolor='black', width=0.4, label='ηc = ηd = 0.95')
plt.bar(np.arange(2019, 2024) + 0.6, results_array_090, color='lightgreen', edgecolor='black', width=0.4, label='ηc = ηd = 0.90')

plt.xlabel('Year (yr)')
plt.ylabel('Aggregated Profit (DKK)')
plt.title('Aggregated Profit for Each Year and Efficiency Case (DKK/yr)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(2019, 2024))
plt.legend()
plt.tight_layout()
plt.show()



#Let us make two linear regressions to see the effects of varying the 
#efficiency vs the average spotprice:


# Computing the effects of varying the efficiency:
efficiencies = np.array([99, 95, 90])
profits_year1 = np.array([669.22003092, 520.0605486, 375.60529113])
profits_year2 = np.array([982.33856184, 855.81117392, 721.46188625])
profits_year3 = np.array([2304.81899261, 1914.01854041, 1501.34619414])
profits_year4 = np.array([5556.79760699, 4632.19670527, 3633.85896278])
profits_year5 = np.array([2569.49584149, 2214.47403819, 1826.19879249])

# Perform linear regression for each year
regressions = []
slopes = []

for profits in [profits_year1, profits_year2, profits_year3, profits_year4, profits_year5]:
    regression = LinearRegression().fit(efficiencies[:, np.newaxis], profits)
    regressions.append(regression)
    slopes.append(regression.coef_[0])

# Compute the average of the slopes
average_slope = np.mean(slopes)
print("Average slope of the relationships:", average_slope)




# Computing the effects of varing the efficiency:
x_values = np.array([669.22003092, 982.33856184, 2304.81899261, 5556.79760699, 2569.49584149])
profits_099 = np.array([669.22003092, 982.33856184, 2304.81899261, 5556.79760699, 2569.49584149])
profits_095 = np.array([520.0605486, 855.81117392, 1914.01854041, 4632.19670527, 2214.47403819])
profits_090 = np.array([375.60529113, 721.46188625, 1501.34619414, 3633.85896278, 1826.19879249])

# Perform linear regression for each year
regressions = []
slopes = []

for profits in [profits_099, profits_095, profits_090]:
    regression = LinearRegression().fit(x_values[:, np.newaxis], profits)
    regressions.append(regression)
    slopes.append(regression.coef_[0])


# Compute the average of the slopes
average_slope = np.mean(slopes)
print("Average slope of the relationships:", average_slope)


#%%

##################################################################
############################ TASK 3 ##############################
##################################################################

"""
Created on Wed Feb 28 15:31:39 2024

@author: lucaslojoiglesias & Jacob Theilgaard
"""

# Import the necessary packages
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import cvxpy as cp
import datetime as dt
from UsefulFunctions import LoadData
from UsefulFunctions import PricesDK
from UsefulFunctions import Netting
from UsefulFunctions import Optimizer
from UsefulFunctions import ProsumerOptimizer


# Since the first tasks uses a changed version of the load function (includes all years) it is here re-inserted:
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
    
    # Keep only 2022 and 2023
    df_prices = df_prices.loc[df_prices["HourDK"].dt.year.isin([2022,2023])]
    
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


### Import price and prosumer data ###
df_prices, df_pro = LoadData()
# Adjust prices with tariffs
df_prices = PricesDK(df_prices)
# Keep only buy/sell prices
df_prices = df_prices[["HourDK","Sell","Buy"]]


##################################### TASK 3.1 ##################################
"""
3.1 For this question consider only consumption and neglect PV generation. Calculate the total costs
of the user for each of the two years (the customer buys energy at the Buy price).
Next, do a rough calculation for comparison. Calculate costs by multiplying the total consumption
of each year with the corresponding average Buy price. Comment on the difference between the
two methods.
"""

# Merge prosumer data with prices on the common 'HourDK' column
df_merged = pd.merge(df_pro, df_prices, on='HourDK', how='inner')

# Calculate the hourly cost for each hour by multiplying consumption with the buy price
df_merged['HourlyCost'] = df_merged['Load'] * df_merged['Buy']

# Extract year from 'HourDK' to separate data for 2022 and 2023
df_merged['Year'] = pd.to_datetime(df_merged['HourDK']).dt.year

# Group by year and sum the hourly costs to get total costs for each year
total_costs_per_year = df_merged.groupby('Year')['HourlyCost'].sum()

# Group by year and sum the consumption to get total consumption for each year
total_consumption_per_year = df_merged.groupby('Year')['Load'].sum()

# Calculate the average buy price for each year
average_buy_price_per_year = df_merged.groupby('Year')['Buy'].mean()

"""
# Saved for manual check
print("Total consumption for 2022: {:.2f} kWh".format(total_consumption_per_year.loc[2022]))
print("Total consumption for 2023: {:.2f} kWh".format(total_consumption_per_year.loc[2023]))

print("Average buy price for 2022: {:.4f} DKK per kWh".format(average_buy_price_per_year.loc[2022]))
print("Average buy price for 2023: {:.4f} DKK per kWh".format(average_buy_price_per_year.loc[2023]))
"""

print("Total costs for 2022: {:.2f} DKK".format(total_costs_per_year.loc[2022]))
print("Total costs for 2023: {:.2f} DKK".format(total_costs_per_year.loc[2023]))


################

# Calculating costs by multiplying total consumption of each year with the corresponding average Buy price
rough_costs_2022 = total_consumption_per_year.loc[2022] * average_buy_price_per_year.loc[2022]
rough_costs_2023 = total_consumption_per_year.loc[2023] * average_buy_price_per_year.loc[2023]

print("\nRough calculation of costs:")
print("Rough 2022: {:.2f} DKK".format(rough_costs_2022))
print("Rough 2023: {:.2f} DKK".format(rough_costs_2023))

# Difference between the two methods
cost_difference_2022 = total_costs_per_year.loc[2022] - rough_costs_2022
cost_difference_2023 = total_costs_per_year.loc[2023] - rough_costs_2023
"""
print("\nDifference between the two methods:")
print("Difference for 2022: {:.2f} DKK".format(cost_difference_2022))
print("Difference for 2023: {:.2f} DKK".format(cost_difference_2023))
"""

##################################### TASK 3.2 ##################################
"""
3.2 Calculate the yearly benefit of the PV system. You can do that by comparing the total costs
without PV generation (which you calculated in 3.1) with the costs of the prosumer under net
metering when owning the PV. Calculate the benefit on a yearly basis and discuss the results.
Remember that under net metering you need to consider imports and exports.
Do you think the PV is a good investment over a 20-year period? You will need some assumptions
to answer the question. You can look up online for the approximate cost of a 5 kW PV system.
"""

# Calculating the yearly benefit of the PV system using the "Netting" function from UsefulFunctions.py

# First, we ensure that the 'Year' column exists in both df_pro and df_prices for grouping
df_prices['Year'] = pd.to_datetime(df_prices['HourDK']).dt.year
df_pro['Year'] = pd.to_datetime(df_pro['HourDK']).dt.year

# Call the Netting function to calculate yearly net profits
yearly_net_profit = Netting(df_pro, df_prices, "Hourly")


# The 'Profit' column in the resulting DataFrame contains the yearly benefit of the PV system
print("Yearly Net Profit with PV:")
print(yearly_net_profit[['Year', 'Profit']])


# Calculate the benefit of having PV installed by comparing the profit with the total costs without PV
yearly_benefit = yearly_net_profit.copy()
for year in yearly_benefit['Year']:
    total_cost_without_pv = total_costs_per_year.loc[year]
    yearly_benefit.loc[yearly_benefit['Year'] == year, 'Benefit'] = yearly_benefit.loc[yearly_benefit['Year'] == year, 'Profit'] + total_cost_without_pv

print("\nYearly Benefit of having PV installed (Profit w PV - Cost <negative> without PV):")
print(yearly_benefit[['Year', 'Benefit']])


##################################### TASK 3.3 ##################################
"""
3.3 Now assume that the prosumer buys the battery described in Task 2. Your charging and discharging efficiencies are equal to 95%. Optimize the operation of the battery for every day of 2022 and
2023, with the goal of minimizing prosumer costs under net metering. What is the benefit that
the battery brings to the prosumer on a yearly basis? The comparison should be made between
the case where the prosumer owns only the PV and the case where both the PV and a battery
are present.
"""


# Define the parameters for the ProsumerOptimizer
battery_params = {
    'Pmax': 5,    # Max power capacity of the battery (kW)
    'Cmax': 10,   # Max energy capacity of the battery (kWh)
    'n_c': 0.95,  # Charging efficiency (95%)
    'n_d': 0.95,  # Discharging efficiency (95%)
    'C_0': 5,     # Initial state of charge in kWh (50% of 10 kWh)
    'C_n': 5      # Final state of charge in kWh (50%)
}


# Extract hourly buy and sell prices
l_b = df_prices['Buy'].values
l_s = df_prices['Sell'].values

# Extracting 'PV' and 'Load' columns with PV generation and load data, respectively
p_PV = df_pro['PV'].values
p_L = df_pro['Load'].values

# Initialize the results dictionary
optimization_results = {
    'Year': [],
    'Total Cost with Battery': [],
    'Total Cost without Battery': [],
    'Benefit with Battery': []
}

# Iterate over each year
for year in [2022, 2023]:
    # Select the year's data
    year_data_prices = df_prices[df_prices['Year'] == year]
    year_data_pro = df_pro[df_pro['Year'] == year]
    
    
    # Optimize the operation of the battery for the year
    cost_with_battery, p_c, p_d, p_b, p_s, X = ProsumerOptimizer(
        battery_params, year_data_prices['Buy'].values, year_data_prices['Sell'].values,
        year_data_pro['PV'].values, year_data_pro['Load'].values
    )
    
    
    # Retrieve the total cost without battery from the Netting function results
    cost_without_battery = yearly_net_profit[yearly_net_profit['Year'] == year]['Profit'].values[0]
    
    # Calculate the benefit of the battery
    benefit_with_battery = cost_with_battery - cost_without_battery
    
    # Store the results
    optimization_results['Year'].append(year)
    optimization_results['Total Cost with Battery'].append(cost_with_battery)
    optimization_results['Total Cost without Battery'].append(cost_without_battery)
    optimization_results['Benefit with Battery'].append(benefit_with_battery)
    
    
    # Status (trying to investigate low values(!!!) - didnt suceed)
    print(f"\nYear: {year}")
    print(f"Total Cost with Battery: {cost_with_battery:.2f}")
    print(f"Total Cost without Battery (from net profit): {cost_without_battery:.2f}")
    print(f"Benefit with Battery: {benefit_with_battery:.2f}")
    


