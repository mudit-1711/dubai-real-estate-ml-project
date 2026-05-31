import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv("area_prices_monthly.csv")
print(df.head())