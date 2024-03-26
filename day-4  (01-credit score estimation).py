# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X = np.array([[600, 25], [700, 35], [750, 45], [620, 30], [800, 50], [680, 40]])
y = np.array([0, 1, 1, 0, 1, 1])  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_instance = np.array([[720, 38]])  
prediction = classifier.predict(new_instance)
print("Predicted class for new instance:", prediction)
