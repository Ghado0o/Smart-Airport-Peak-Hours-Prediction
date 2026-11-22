import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

print("Loading Data...")

df = pd.read_csv("bts_1995_2026_clean.csv")

df = df.sample(
    n=500000,
    random_state=42
)

features = [
    "YEAR",
    "MONTH",
    "AIRLINE_ID",
    "ORIGIN_AIRPORT_ID",
    "DEST_AIRPORT_ID",
    "DISTANCE_GROUP"
]

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=features + ["PASSENGERS"]
)

print("Missing Values:")
print(df[features + ["PASSENGERS"]].isnull().sum())

X = df[features]
y = df["PASSENGERS"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training XGBoost...")

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nXGBoost Results")
print("MAE =", mean_absolute_error(y_test, pred))
print("R2 =", r2_score(y_test, pred))