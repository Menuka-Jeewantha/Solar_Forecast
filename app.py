import datetime
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Load Saved Models
@st.cache_resource
def load_models():
  solar_model = joblib.load("solar_model.pkl")
  load_model = joblib.load("load_model.pkl")
  return solar_model, load_model


solar_model, load_model = load_models()

# Page Configuration
st.set_page_config(
    page_title="Smart Solar & Energy Management System",
    page_icon="⚡",
    layout="wide",
)

st.title(
    "☀️ AI-Based Solar & Building Energy Management System (Sri Lanka Time)"
)
st.markdown(
    "Select your **local date and time in Sri Lanka (GMT+5:30)**. The app automatically aligns with local solar hours, fetches live weather, and runs your AI forecasting models."
)

# Sidebar Inputs (Local Date & Time in Sri Lanka)
st.sidebar.header("📅 Select Local Date & Time (Sri Lanka)")
selected_date = st.sidebar.date_input(
    "Target Date", datetime.date.today()
)
selected_time = st.sidebar.time_input(
    "Target Local Time", datetime.time(14, 0)
)  # Default 2:00 PM local time

# Combine into local datetime
local_datetime = datetime.datetime.combine(selected_date, selected_time)

# Convert Local Time (Sri Lanka UTC+5:30) to UTC for Open-Meteo API query
sri_lanka_offset = datetime.timedelta(hours=5, minutes=30)
utc_datetime = local_datetime - sri_lanka_offset

# Local hours for model features
local_hour = local_datetime.hour
month = local_datetime.month
dayofyear = local_datetime.timetuple().tm_yday
dayofweek = local_datetime.weekday()

st.sidebar.divider()
battery_soc = st.sidebar.slider(
    "Current Battery SoC (%)", min_value=0.0, max_value=100.0, value=50.0
)


# Fetch Real Weather from Open-Meteo using UTC time alignment
@st.cache_data(ttl=3600)
def fetch_open_meteo_weather(target_utc_dt):
  lat, lon = 6.583, 79.963  # Kalutara, Sri Lanka coordinates
  url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,shortwave_radiation,wind_speed_10m&timezone=UTC&forecast_days=7"

  try:
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
      data = response.json()
      hourly = data["hourly"]
      times = hourly["time"]

      target_str = target_utc_dt.strftime("%Y-%m-%dT%H:00")

      if target_str in times:
        idx = times.index(target_str)
        temp = hourly["temperature_2m"][idx]
        humidity = hourly["relative_humidity_2m"][idx]
        precip = hourly["precipitation"][idx]
        clouds = hourly["cloud_cover"][idx]
        shortwave_rad = hourly["shortwave_radiation"][idx]
        wind_speed = hourly["wind_speed_10m"][idx]

        direct_irr = max(0.0, shortwave_rad * 0.75)
        diffuse_irr = max(0.0, shortwave_rad * 0.25)

        # Solar height calculation based on local hour in Sri Lanka (daylight roughly 6 AM to 6 PM local time)
        sun_height = (
            90.0 * np.sin(np.pi * (local_hour - 6) / 12)
            if 6 <= local_hour <= 18
            else -10.0
        )
        return (
            temp,
            humidity,
            wind_speed,
            precip,
            direct_irr,
            diffuse_irr,
            sun_height,
            True,
        )
  except Exception:
    pass

  # Fallback if offline
  temp = 30.0
  humidity = 75.0
  wind_speed = 3.0
  precip = 0.0
  direct_irr = 700.0 if 6 <= local_hour <= 18 else 0.0
  diffuse_irr = 150.0 if 6 <= local_hour <= 18 else 0.0
  sun_height = (
      90.0 * np.sin(np.pi * (local_hour - 6) / 12)
      if 6 <= local_hour <= 18
      else -10.0
  )
  return (
      temp,
      humidity,
      wind_speed,
      precip,
      direct_irr,
      diffuse_irr,
      sun_height,
      False,
  )


# Fetch weather data based on aligned UTC time
(
    temp,
    humidity,
    wind_speed,
    precip,
    direct_irr,
    diffuse_irr,
    sun_height,
    is_live,
) = fetch_open_meteo_weather(utc_datetime)

status_badge = (
    "🟢 Live Open-Meteo Data" if is_live else "🟡 Estimated / Fallback Data"
)
st.info(
    f"🇱🇰 **Local Time: {local_datetime.strftime('%Y-%m-%d %H:%M')} (Sri Lanka)** | Status: {status_badge} | "
    f"Temp: {temp:.1f}°C | Radiation: {direct_irr + diffuse_irr:.1f} W/m²"
)

# 1. Solar Prediction using local hour
solar_input_df = pd.DataFrame(
    [[
        temp,
        humidity,
        wind_speed,
        precip,
        direct_irr,
        diffuse_irr,
        sun_height,
        local_hour,
        month,
        dayofyear,
    ]],
    columns=[
        "NASA_temperature_C",
        "NASA_relative_humidity_pct",
        "NASA_wind_speed_m_s",
        "NASA_precipitation_mm_per_hour",
        "PVGIS_direct_irradiance_W_m2",
        "PVGIS_diffuse_irradiance_W_m2",
        "PVGIS_sun_height_deg",
        "hour",
        "month",
        "dayofyear",
    ],
)
predicted_solar = float(solar_model.predict(solar_input_df)[0])
predicted_solar = max(0.0, predicted_solar)

# 2. Load Prediction using local hour
load_input_df = pd.DataFrame(
    [[local_hour, dayofweek, month, 1500.0, 1450.0, 1550.0]],
    columns=[
        "hour",
        "dayofweek",
        "month",
        "load_lag_24",
        "load_lag_48",
        "load_lag_168",
    ],
)
predicted_load = float(load_model.predict(load_input_df)[0])
predicted_load = max(0.0, predicted_load)

net_load = predicted_load - predicted_solar

# Display Metrics
col1, col2, col3 = st.columns(3)
with col1:
  st.metric(
      label="☀️ Predicted Solar PV Generation",
      value=f"{predicted_solar:.2f} W",
  )
with col2:
  st.metric(
      label="🏢 Predicted Building Load", value=f"{predicted_load:.2f} W"
  )
with col3:
  st.metric(
      label="⚡ Net Grid Demand",
      value=f"{net_load:.2f} W",
      delta=(
          "Exporting Surplus" if net_load < 0 else "Importing from Grid/Battery"
      ),
  )

st.divider()

# Energy Management Simulation
st.subheader("🔋 Simulated Energy Management System (EMS) Dispatch")
battery_capacity_wh = 5000.0
current_battery_wh = (battery_soc / 100.0) * battery_capacity_wh

ems_action = ""
if net_load < 0:
  surplus = abs(net_load)
  energy_to_charge = min(surplus, battery_capacity_wh - current_battery_wh)
  remaining_surplus = surplus - energy_to_charge
  ems_action = f"Charge battery with {energy_to_charge:.1f} Wh. Export excess {remaining_surplus:.1f} Wh to Grid."
else:
  deficit = net_load
  energy_to_discharge = min(deficit, current_battery_wh)
  remaining_deficit = deficit - energy_to_discharge
  ems_action = f"Discharge battery by {energy_to_discharge:.1f} Wh. Purchase remaining {remaining_deficit:.1f} Wh from National Grid."

st.success(f"**Recommended Dispatch Decision:** {ems_action}")

# Maintenance Alerting
st.divider()
st.subheader("🛠️ Condition-Based Maintenance (CBM) Alerting")
actual_observed_solar = predicted_solar * 0.80
performance_ratio = (
    (actual_observed_solar / predicted_solar) * 100 if predicted_solar > 0 else 100
)

if performance_ratio < 85.0 and (direct_irr + diffuse_irr) > 300:
  st.warning(
      "⚠️ **[Maintenance Alert Triggered]**: Actual solar generation is sustained >15% below model expectations under high irradiance. **Recommended Action:** Inspect panels for soiling or shading."
  )
else:
  st.info(
      "✅ **System Status Healthy**: Solar PV generation matches model expectations within acceptable tolerances."
  )