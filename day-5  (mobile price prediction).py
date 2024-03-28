import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("C:\machine learning\mobile data.csv")

X = data[['battery_power', 'ram', 'talk_time']] 
y = data['price_range'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
new_mobile_features = [[1500, 2048, 10]] 
predicted_price = model.predict(new_mobile_features)
print("Predicted price for the new mobile:", predicted_price)
