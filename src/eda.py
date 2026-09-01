import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/cleaned_weather.csv")

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Create graphs folder
os.makedirs("graphs", exist_ok=True)

# 1. Temperature distribution
plt.figure(figsize=(10, 6))

plt.hist(df["Temperature_Avg (°C)"], bins=30)

plt.xlabel("Average Temperature (°C)")
plt.ylabel("Frequency")
plt.title("Temperature Distribution")

plt.tight_layout()
plt.savefig("graphs/temperature_distribution.png")
plt.show()


# 2. Weather condition distribution
# Your dataset does not have WeatherCondition,
# so use AQI_Category instead.
plt.figure(figsize=(10, 6))

df["AQI_Category"].value_counts().plot(kind="bar")

plt.xlabel("AQI Category")
plt.ylabel("Count")
plt.title("AQI Category Distribution")

plt.tight_layout()
plt.savefig("graphs/aqi_categories.png")
plt.show()


# 3. Temperature over time
plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Temperature_Avg (°C)"]
)

plt.xlabel("Date")
plt.ylabel("Average Temperature (°C)")
plt.title("Temperature Over Time")

plt.tight_layout()
plt.savefig("graphs/temperature_over_time.png")
plt.show()


# 4. Correlation heatmap
plt.figure(figsize=(12, 8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Weather Parameter Correlation")

plt.tight_layout()
plt.savefig("graphs/correlation.png")
plt.show()

print("EDA completed successfully!")
print("Graphs saved in the 'graphs' folder.")