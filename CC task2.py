# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load historical sales data into a DataFrame
sales_data = pd.read_csv('/content/sales_data.csv.csv')

# Convert 'Date' column to datetime format and set the correct date format
sales_data['Date'] = pd.to_datetime (sales_data['Date'], format='%d/%m/%Y', dayfirst=True)

# Set 'Date' column as the index
sales_data.set_index('Date', inplace=True)

# Specify frequency as daily ('D')
sales_data.index.freq = 'D'

# Visualize the sales data
plt.figure(figsize=(10, 6))
plt.plot(sales_data.index, sales_data['Store'], marker='o', linestyle='-')
plt.title('Store-wise Sales Data')
plt.xlabel('Date')
plt.ylabel('Store')
plt.grid(True)
plt.show()

# Decompose the time series to analyze trend, seasonality, and residuals
decomposition = seasonal_decompose(sales_data['Store'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Train-test split
train_size = int(len(sales_data) * 0.8)
train, test = sales_data.iloc[:train_size], sales_data.iloc[train_size:]

# Fit Holt-Winters Exponential Smoothing model
model = ExponentialSmoothing(train['Store'], seasonal_periods=12, trend='add', seasonal='add').fit()

# Forecast
forecast = model.forecast(len(test))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test['Store'], forecast))
print(f'RMSE: {rmse}')

# Visualize forecast
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Store'], label='Training Data', color='blue')
plt.plot(test.index, test['Store'], label='Test Data', color='green')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('Forecasting Store Demand')
plt.xlabel('Date')
plt.ylabel('Store')
plt.legend(loc='best')
plt.grid(True)
plt.show()
