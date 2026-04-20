import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load CSV file
# Replace with your actual file name
file_path = "flight_data_2024.csv"
df = pd.read_csv(
    file_path,
    usecols=["month", "day_of_month", "dep_delay"]
)
# --- Clean the data ---
# Remove missing delays
df = df.dropna(subset=["dep_delay"])

# --- Create a time column ---
# Combine month + day into a single sortable value
df["date"] = pd.to_datetime(
    df.rename(columns={"day_of_month": "day"})[["month", "day"]].assign(year=2024)
)

# --- Group and compute average delay ---
avg_delay = df.groupby("date")["dep_delay"].mean()

# --- Plot ---
plt.figure(figsize=(12,6))
plt.plot(avg_delay.index, avg_delay.values)

plt.title("Average Departure Delay vs Time")
plt.xlabel("Date")
plt.ylabel("Average Delay (minutes)")

# --- Set monthly ticks --- 
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Jan, Feb, etc.

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
