import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("area_prices_monthly.csv")

# remove target
y = df['secondary_price_per_sqft_usd']

# features (everything except target + optional leakage columns)
X = df.drop(columns=[
    'secondary_price_per_sqft_usd',
    'secondary_price_per_m2_usd'
])

# handle categorical columns
X = pd.get_dummies(X)

# fill missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# model
model = LinearRegression()
model.fit(X_train, y_train)

print("R2:", model.score(X_test, y_test))

# save model
pickle.dump(model, open("model.pkl", "wb"))

# IMPORTANT: save feature columns for frontend
pickle.dump(X.columns, open("columns.pkl", "wb"))
print("TRAINING STARTED")

print(X.columns)
print("Model trained successfully")