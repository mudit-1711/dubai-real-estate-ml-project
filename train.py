import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("area_prices_monthly.csv")

# ONLY these columns
X = df[['area_size', 'beds', 'baths']]
y = df['secondary_price_per_sqft_usd']

# Fill missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Save NEW model
pickle.dump(model, open('model.pkl', 'wb'))

print("NEW model saved")

print(X.columns)