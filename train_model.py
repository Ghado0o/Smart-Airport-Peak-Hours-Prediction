import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

print("Loading dataset...")

df = pd.read_csv(
    "bts_1995_2026_clean.csv"
)

# استخدام عينة لتسريع التدريب
df = df.sample(
    n=500000,
    random_state=42
)

df = df.dropna(
    subset=[
        "YEAR",
        "MONTH",
        "DISTANCE_GROUP",
        "PASSENGERS"
    ]
)

print(df.columns.tolist())

features = [
    "YEAR",
    "MONTH",
    "AIRLINE_ID",
    "ORIGIN_AIRPORT_ID",
    "DEST_AIRPORT_ID",
    "DISTANCE_GROUP"
]

X = df[features]

y = df["PASSENGERS"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nResults")
print("MAE =", mean_absolute_error(y_test, pred))
print("R2 =", r2_score(y_test, pred))

#اهم المتغيرات المؤثرة على عدد الركاب

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.bar(
    importance["Feature"],
    importance["Importance"],
    color="steelblue"
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

# حساب الحدود الإحصائية من البيانات

q25 = df["PASSENGERS"].quantile(0.25)
q75 = df["PASSENGERS"].quantile(0.75)

print("\nQuartile Thresholds")
print("Q25 =", round(q25, 2))
print("Q75 =", round(q75, 2))

#تحويل المشروع من نظام تنبؤ إلى نظام إدارة الذروة ذكي

def manage_peak(predicted_passengers):

    if predicted_passengers < q25:

        level = "LOW"

        actions = [
            "Normal Operations",
            "Standard Staffing",
            "Regular Check-in Counters"
        ]

    elif predicted_passengers < q75:

        level = "MEDIUM"

        actions = [
            "Increase Check-in Counters",
            "Add Security Staff",
            "Prepare Extra Gates"
        ]

    else:

        level = "HIGH"

        actions = [
            "Open Additional Gates",
            "Increase Security Teams",
            "Increase Check-in Counters",
            "Deploy Passenger Guidance Staff",
            "Activate Peak Management Plan"
        ]

    return level, actions

    print(importance)
    # =========================
# Peak Management System
# =========================

future_flight = X_test.iloc[[0]]

predicted_passengers = model.predict(future_flight)[0]

print("\n" + "="*50)
print("Predicted Passengers:", round(predicted_passengers))
print("="*50)

level, actions = manage_peak(predicted_passengers)

import matplotlib.pyplot as plt

levels = ["LOW", "MEDIUM", "HIGH"]
values = [0, 0, 0]

if level == "LOW":
    values = [100, 0, 0]

elif level == "MEDIUM":
    values = [0, 100, 0]

else:
    values = [0, 0, 100]

plt.figure(figsize=(8,5))

bars = plt.bar(
    levels,
    values,
    color=["green", "orange", "red"]
)

plt.title("Peak Level Classification")
plt.ylabel("Probability (%)")

for bar in bars:
    height = bar.get_height()

    if height > 0:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 2,
            f"{height:.0f}%",
            ha="center"
        )

plt.savefig("peak_level_classification.png")

plt.show()

# ===================================
# Peak Classification Comparison
# ===================================

