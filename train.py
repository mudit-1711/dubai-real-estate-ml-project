import pandas as pd 
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

df = pd.read_csv("area_prices_monthly.csv")
X = df.drop(columns=['secondary_price_per_sqft_usd'])
y = df['secondary_price_per_sqft_usd']

X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

my_model = LinearRegression()

my_model.fit(X_train, y_train)

y_pred = my_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
pickle.dump(my_model, open('model.pkl', 'wb'))
print("model saved")
