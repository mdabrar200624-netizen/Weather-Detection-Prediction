import pandas as pd

# Load dataset
df = pd.read_csv("data/weather.csv")

print("Original shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Display missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill numerical missing values with median
numeric_columns = [
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Rainfall"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = df[column].fillna(df[column].median())

# Remove rows where weather condition is missing
if "WeatherCondition" in df.columns:
    df = df.dropna(subset=["WeatherCondition"])

print("\nCleaned shape:", df.shape)

# Save cleaned dataset
df.to_csv("data/cleaned_weather.csv", index=False)

print("\nCleaned dataset saved successfully.")