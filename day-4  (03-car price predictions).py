import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("C:\machine learning\CarPrice.csv") 

X = data[['wheelbase', 'enginesize', 'horsepower']]  
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

new_data_point = [[50000, 5, 200]]  
predicted_price = model.predict(new_data_point)
print("Predicted price for the new data point:", predicted_price)
