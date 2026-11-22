import pandas as pd

df = pd.read_csv("bts_1995_2026_clean.csv")

print("=" * 50)
print("Shape:")
print(df.shape)

print("=" * 50)
print("Columns:")
print(df.columns.tolist())

print("=" * 50)
print("Statistics:")
print(df.describe())

print("=" * 50)
print("Top Years:")
print(df["YEAR"].value_counts().sort_index())

print("=" * 50)
print("Top Months:")
print(df["MONTH"].value_counts().sort_index())