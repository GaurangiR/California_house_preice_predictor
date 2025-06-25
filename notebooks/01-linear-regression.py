import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("C:\\Users\\gaura\\Desktop\\ML\\LinearRegression-HousePrices\\data\\housing.csv")
print(data.head())
print(data.info())

# Drop missing values
data.dropna(inplace=True)
print(data.info())

# Train-test split
from sklearn.model_selection import train_test_split
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Combine into a train DataFrame
train_data = X_train.join(y_train)

# Log transform to normalize skewed distributions
for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
    train_data[col] = train_data[col].clip(lower=1)  # avoid log(0)
    train_data[col] = np.log(train_data[col])

# One-hot encode ocean_proximity
train_data = train_data.join(pd.get_dummies(train_data['ocean_proximity']))
train_data.drop("ocean_proximity", axis=1, inplace=True)

# Feature engineering
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

# Prepare features and target
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X_train = train_data.drop("median_house_value", axis=1)
y_train = train_data["median_house_value"]
feature_columns = X_train.columns  # Save column names for test alignment

# Clean & scale
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.dropna(inplace=True)
y_train = y_train.loc[X_train.index]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# ---------- Test Set Processing ----------
test_data = X_test.join(y_test)

# Log transform on test set
for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
    test_data[col] = test_data[col].clip(lower=1)
    test_data[col] = np.log(test_data[col])

# One-hot encode test ocean_proximity
test_data = test_data.join(pd.get_dummies(test_data['ocean_proximity']))
test_data.drop("ocean_proximity", axis=1, inplace=True)

# Feature engineering for test
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

# Add missing dummy columns (if any)
for col in feature_columns:
    if col not in test_data.columns:
        test_data[col] = 0

# Reorder columns to match training
test_data = test_data[feature_columns]

# Clean & scale test data
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data.dropna(inplace=True)
y_test = y_test.loc[test_data.index]
X_test = scaler.transform(test_data)

# Score the model
score = reg.score(X_test, y_test)
print(f"Test R² Score: {score:.4f}")
import os
import joblib

# Create 'models' directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model and scaler
joblib.dump(reg, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

