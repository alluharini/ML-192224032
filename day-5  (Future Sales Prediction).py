# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Sample data (you should replace this with your actual sales data)
data = pd.Series([100 + i + (i % 7) for i in range(365)], 
                 index=pd.date_range(start='2023-01-01', periods=365))

# Fit SARIMA model
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Make future predictions
forecast = results.get_forecast(steps=30)  # Predicting sales for the next 30 days

# Plotting the forecast
plt.figure(figsize=(10, 6))
data.plot(label='Actual Sales')
forecast.predicted_mean.plot(label='Forecasted Sales')
plt.title('Future Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
