import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

X = df[features]
y = df["PASSENGERS"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

print("Training ANN...")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=256,
    verbose=1
)

pred = model.predict(X_test)

print("\nANN Results")
print("MAE =", mean_absolute_error(y_test, pred))
print("R2 =", r2_score(y_test, pred))