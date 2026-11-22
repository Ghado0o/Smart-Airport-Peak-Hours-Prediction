import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bts_1995_2026_clean.csv")

# مجموع الركاب لكل سنة
yearly = df.groupby("YEAR")["PASSENGERS"].sum()

plt.figure(figsize=(12,6))
yearly.plot()
plt.title("Total Passengers by Year")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.grid(True)

plt.savefig("passengers_by_year.png")
plt.close()
#حلل لافضل السنوات
top_years = (
    df.groupby("YEAR")["PASSENGERS"]
    .sum()
    .sort_values(ascending=False)
)

print(top_years.head(10))