import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/cleaned_weather.csv")

print("Dataset loaded successfully.")
print(f"Total records: {len(df)}")


# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Extract date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfYear"] = df["Date"].dt.dayofyear

# Seasonal/cyclical features
df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)

df["DayOfYear_Sin"] = np.sin(
    2 * np.pi * df["DayOfYear"] / 365.25
)

df["DayOfYear_Cos"] = np.cos(
    2 * np.pi * df["DayOfYear"] / 365.25
)


# Features
numeric_features = [
    "Humidity (%)",
    "Rainfall (mm)",
    "Wind_Speed (km/h)",
    "Pressure (hPa)",
    "Cloud_Cover (%)",
    "Year",
    "Month",
    "Day",
    "DayOfYear",
    "Month_Sin",
    "Month_Cos",
    "DayOfYear_Sin",
    "DayOfYear_Cos"
]

categorical_features = [
    "City"
]

features = numeric_features + categorical_features

X = df[features]

# Target
y = df["Temperature_Avg (°C)"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "city",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Random Forest
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)


# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train
print("\nTraining temperature prediction model...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# Predict
y_pred = pipeline.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("\n======================================")
print("Temperature Prediction Results")
print("======================================")

print(f"Mean Absolute Error : {mae:.2f} °C")
print(f"Mean Squared Error : {mse:.2f}")
print(f"Root Mean Squared Error : {rmse:.2f} °C")
print(f"R² Score : {r2:.2f}")


# Sample predictions
results = pd.DataFrame({
    "Actual Temperature": y_test.values[:10],
    "Predicted Temperature": y_pred[:10]
})

print("\nSample Predictions:")
print(results.to_string(index=False))


# Save model
joblib.dump(
    pipeline,
    "models/temperature_prediction_model.pkl"
)

print("\nModel saved successfully.")
print("models/temperature_prediction_model.pkl")