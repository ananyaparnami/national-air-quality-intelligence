import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import xgboost as xgb

DB_URI = "postgresql://username:password@your-neon-host/neondb?sslmode=require"
engine = create_engine(DB_URI)

print("1/4: Reading city_day.csv...")
df = pd.read_csv("city_day.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["AQI", "City"]).sort_values(["City", "Date"])

print("2/4: Pushing historical cleaned data to Neon Cloud...")
df.to_sql("city_day_clean", engine, if_exists="replace", index=False)

forecast_records = []
unique_cities = df["City"].unique()

print(f"3/4: Training XGBoost ML models for {len(unique_cities)} cities...")
for city in unique_cities:
  city_data = df[df["City"] == city].copy()
  if len(city_data) < 30:
    continue

  city_data["Lag_1"] = city_data["AQI"].shift(1)
  city_data["Lag_7"] = city_data["AQI"].shift(7)
  city_data["Rolling_Mean_7"] = (
      city_data["AQI"].shift(1).rolling(window=7, min_periods=1).mean()
  )
  city_data["DayOfWeek"] = city_data["Date"].dt.dayofweek
  city_data["Month"] = city_data["Date"].dt.month

  train_set = city_data.dropna()
  features = ["Lag_1", "Lag_7", "Rolling_Mean_7", "DayOfWeek", "Month"]

  X = train_set[features]
  y = train_set["AQI"]

  model = xgb.XGBRegressor(
      n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
  )
  model.fit(X, y)

  last_date = city_data["Date"].max()
  last_window = list(city_data["AQI"].tail(7).values)

  for day_offset in range(1, 8):
    target_date = last_date + datetime.timedelta(days=day_offset)
    lag_1 = last_window[-1]
    lag_7 = last_window[0] if len(last_window) >= 7 else last_window[-1]
    roll_7 = np.mean(last_window[-7:])
    dow = target_date.weekday()
    month = target_date.month

    feature_vec = pd.DataFrame(
        [[lag_1, lag_7, roll_7, dow, month]], columns=features
    )
    predicted_val = float(model.predict(feature_vec)[0])
    predicted_val = max(10.0, round(predicted_val, 2))

    if predicted_val <= 50:
      bucket = "Good"
    elif predicted_val <= 100:
      bucket = "Satisfactory"
    elif predicted_val <= 200:
      bucket = "Moderate"
    elif predicted_val <= 300:
      bucket = "Poor"
    elif predicted_val <= 400:
      bucket = "Very Poor"
    else:
      bucket = "Severe"

    lower_ci = round(predicted_val * 0.88, 2)
    upper_ci = round(predicted_val * 1.12, 2)

    forecast_records.append({
        "City": city,
        "Forecast_Date": target_date.strftime("%Y-%m-%d"),
        "Predicted_AQI": predicted_val,
        "Lower_CI_95": lower_ci,
        "Upper_CI_95": upper_ci,
        "Forecast_Category": bucket,
    })

    last_window.append(predicted_val)

print("4/4: Uploading aqi_forecast_7days table to Cloud DB...")
forecast_df = pd.DataFrame(forecast_records)
forecast_df.to_sql("aqi_forecast_7days", engine, if_exists="replace", index=False)
print("SUCCESS: Pipeline completed and database synced!")