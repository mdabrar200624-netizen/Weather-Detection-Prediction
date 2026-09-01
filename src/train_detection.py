import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/cleaned_weather.csv")

print("Dataset loaded successfully.")
print(f"Total records: {len(df)}")


# ============================================================
# CREATE WEATHER CONDITION
# ============================================================

def determine_weather(row):

    if row["Rainfall (mm)"] >= 10:
        return "Rainy"

    elif row["Cloud_Cover (%)"] >= 70:
        return "Cloudy"

    elif row["Humidity (%)"] >= 75:
        return "Humid"

    else:
        return "Sunny"


df["WeatherCondition"] = df.apply(
    determine_weather,
    axis=1
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "Temperature_Max (°C)",
    "Temperature_Min (°C)",
    "Temperature_Avg (°C)",
    "Humidity (%)",
    "Rainfall (mm)",
    "Wind_Speed (km/h)",
    "Pressure (hPa)",
    "Cloud_Cover (%)"
]

X = df[features]


# ============================================================
# TARGET
# ============================================================

y = df["WeatherCondition"]


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nWeather Condition Distribution:")
print(y.value_counts())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining weather detection model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n======================================")
print("Weather Detection Results")
print("======================================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": model.feature_importances_
    }
).sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/weather_detection_model.pkl"
)

print("\nModel saved successfully.")
print(
    "models/weather_detection_model.pkl"
)