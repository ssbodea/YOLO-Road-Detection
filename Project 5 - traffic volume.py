import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================
# LOAD DATA
# ============================
df = pd.read_csv(r"D:\SBC_PROIECT\mitv.csv")

# Convert date/time
df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M")
df["hour"] = df["date_time"].dt.hour
df["dayofweek"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month
df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

# One-hot encode
df = pd.get_dummies(df, columns=["holiday", "weather_main", "weather_description"], drop_first=True)

# Features/labels
X = df.drop(["traffic_volume", "date_time"], axis=1)
y = df["traffic_volume"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# TRAIN MODELS
# ============================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

colors = {
    "Linear Regression": "blue",
    "Random Forest": "green",
    "XGBoost": "red",
    "Actual": "black"
}

predictions = {}
results = []

# Train and evaluate all models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions[name] = preds
    
    # Metrics
    MAE = mean_absolute_error(y_test, preds)
    RMSE = np.sqrt(mean_squared_error(y_test, preds))
    R2 = r2_score(y_test, preds)
    results.append((name, MAE, RMSE, R2))

# ============================
# SHOW RESULTS
# ============================
print("\n=== MODEL COMPARISON ===")
for r in results:
    print(f"{r[0]:<20} | MAE: {r[1]:.2f} | RMSE: {r[2]:.2f} | R²: {r[3]:.4f}")

# ============================
# 1. SEPARATE GRAPHS FOR EACH MODEL
# ============================
for name, preds in predictions.items():
    plt.figure(figsize=(12,6))
    plt.plot(y_test.values[:300], color="black", label="Actual", linewidth=3)
    plt.plot(preds[:300], color=colors[name], label=name, linewidth=2)
    plt.title(f"Actual vs Predicted — {name}")
    plt.xlabel("Sample Index (subset of test set)")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================
# 2. COMPARISON GRAPH (ALL MODELS + ORIGINAL)
# ============================
plt.figure(figsize=(14,7))
plt.plot(y_test.values[:300], color="black", label="Actual", linewidth=3)

plt.plot(predictions["Linear Regression"][:300], color="blue", label="Linear Regression")
plt.plot(predictions["Random Forest"][:300], color="green", label="Random Forest")
plt.plot(predictions["XGBoost"][:300], color="red", label="XGBoost")

plt.title("Actual vs Predictions — All Models (First 300 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)
plt.show()