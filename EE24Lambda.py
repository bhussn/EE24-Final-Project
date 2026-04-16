import pandas as pd

file_path = "flight_data_2024.csv"
major_delay_threshold = 60

df = pd.read_csv(
    file_path,
    usecols=["month", "day_of_month", "dep_delay"],
    low_memory=False
)

df = df.dropna(subset=["month", "day_of_month", "dep_delay"]).copy()
df["month"] = pd.to_numeric(df["month"], errors="coerce")
df["day_of_month"] = pd.to_numeric(df["day_of_month"], errors="coerce")
df["dep_delay"] = pd.to_numeric(df["dep_delay"], errors="coerce")
df = df.dropna(subset=["month", "day_of_month", "dep_delay"]).copy()

df["month"] = df["month"].astype(int)
df["day_of_month"] = df["day_of_month"].astype(int)
df["is_major_delay"] = df["dep_delay"] >= major_delay_threshold

# Count major delays per day
daily_counts = (
    df.groupby(["month", "day_of_month"])["is_major_delay"]
      .sum()
      .reset_index(name="major_delay_count")
)

# Monthly lambda estimated as average daily count * number of days observed in that month
monthly_summary = (
    daily_counts.groupby("month")["major_delay_count"]
    .agg(["mean", "count", "sum"])
    .rename(columns={"mean": "avg_daily_major_delays",
                     "count": "days_observed",
                     "sum": "lambda_major_monthly"})
    .reset_index()
)

print(monthly_summary)