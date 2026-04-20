import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD (from kagglehub)
import kagglehub

path = kagglehub.dataset_download("hrishitpatil/flight-data-2024")
df = pd.read_csv(path + "/flight_data_2024.csv")

# 2. CLEAN + DEFINE EVENT
df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")
df["dep_delay"] = pd.to_numeric(df["dep_delay"], errors="coerce")

df = df.dropna(subset=["fl_date", "dep_delay"])

# Define Poisson event: major delay
threshold = 60
df["is_event"] = df["dep_delay"] >= threshold

# 3. DAILY COUNT TIME SERIES
daily = df.groupby("fl_date")["is_event"].sum().sort_index()


# 1. INDEPENDENCE TEST (LAG CORRELATION)
print("\n🔹 Independence Test (Lag Correlation)")

for lag in [1, 2, 3, 7]:
    corr = daily.corr(daily.shift(lag))
    print(f"Lag {lag}: {corr:.4f}")

# 2. CONSTANT RATE (MONTHLY CHECK)
df["month"] = df["fl_date"].dt.month

print("\n🔹 Constant Rate Check (Sample Months)")

for m in [4, 7]:
    temp = df[df["month"] == m].groupby("fl_date")["is_event"].sum()

    print(f"\nMonth {m}")
    print("Mean:", temp.mean())
    print("Variance:", temp.var())

    plt.figure(figsize=(8,4))
    plt.plot(temp.values)
    plt.title(f"Daily Major Delays - Month {m}")
    plt.xlabel("Day Index")
    plt.ylabel("Count")
    plt.grid()
    plt.show()


# 3. RARITY CHECK
print("\n🔹 Rarity of Events")

total_flights = len(df)
total_events = df["is_event"].sum()

print("Total flights:", total_flights)
print("Major delay events:", total_events)
print("Event rate:", total_events / total_flights)


# 4. POISSON CHECK (MEAN ≈ VARIANCE)
print("\n🔹 Poisson Check (Mean vs Variance)")

print("Mean:", daily.mean())
print("Variance:", daily.var())
