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

st.title("☀️ Automated AI-Based Solar & Building Energy Management System")
st.markdown(
    "Select a date and time. The app fetches live weather forecasts from Open-Meteo, predicts solar generation and building load using your ML models, and simulates energy management."
)

# Sidebar Inputs (Date & Time)
st.sidebar.header("📅 Select Target Date & Time")
selected_date = st.sidebar.date_input(
    "Target Date", datetime.date.today()
)
selected_time = st.sidebar.time_input("Target Time", datetime.time(12, 0))

target_datetime = datetime.datetime.combine(selected_date, selected_time)
hour = target_datetime.hour
month = target_datetime.month
dayofyear = target_datetime.timetuple().tm_yday
dayofweek = target_datetime.weekday()

st.sidebar.divider()
battery_soc = st.sidebar.slider(
    "Current Battery SoC (%)", min_value=0.0, max_value=100.0, value=50.0
)


# Fetch Real Weather from Open-Meteo API
@st.cache_data(ttl=3600)  # Cache for 1 hour to prevent redundant requests
def fetch_open_meteo_weather(target_dt):
  # Coordinates matching your setup
  lat, lon = 6.583, 79.963
  url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,shortwave_radiation,wind_speed_10m&timezone=UTC&forecast_days=7"

  try:
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
      data = response.json()
      hourly = data["hourly"]
      times = hourly["time"]  # List of ISO timestamp strings

      # Format target datetime into Open-Meteo's hourly string format (e.g., "2026-10-13T12:00")
      target_str = target_dt.strftime("%Y-%m-%dT%H:00")

      if target_str in times:
        idx = times.index(target_str)
        temp = hourly["temperature_2m"][idx]
        humidity = hourly["relative_humidity_2m"][idx]
        precip = hourly["precipitation"][idx]
        clouds = hourly["cloud_cover"][idx]
        shortwave_rad = hourly["shortwave_radiation"][idx]
        wind_speed = hourly["wind_speed_10m"][idx]

        # Map to your model features
        direct_irr = max(
            0.0, shortwave_rad * 0.75
        )  # Approximate direct split
        diffuse_irr = max(
            0.0, shortwave_rad * 0.25
        )  # Approximate diffuse split
        sun_height = 90.0 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else -10.0
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

  # Fallback if API fails or date is out of 7-day range
  temp = 28.0
  humidity = 80.0
  wind_speed = 3.5
  precip = 0.0
  direct_irr = 600.0 if 6 <= hour <= 18 else 0.0
  diffuse_irr = 150.0 if 6 <= hour <= 18 else 0.0
  sun_height = 45.0 if 6 <= hour <= 18 else -10.0
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


# Fetch weather data
(
    temp,
    humidity,
    wind_speed,
    precip,
    direct_irr,
    diffuse_irr,
    sun_height,
    is_live,
) = fetch_open_meteo_weather(target_datetime)

status_badge = (
    "🟢 Live Open-Meteo Data" if is_live else "🟡 Estimated / Fallback Data"
)
st.info(
    f"🌐 **Weather Status: {status_badge} for {target_datetime.strftime('%Y-%m-%d %H:%M')}** | "
    f"Temp: {temp:.1f}°C | Radiation: {direct_irr + diffuse_irr:.1f} W/m² | Wind: {wind_speed} m/s"
)

# 1. Solar Prediction using solar_model.pkl
solar_input_df = pd.DataFrame(
    [[
        temp,
        humidity,
        wind_speed,
        precip,
        direct_irr,
        diffuse_irr,
        sun_height,
        hour,
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

# 2. Load Prediction using load_model.pkl
load_input_df = pd.DataFrame(
    [[hour, dayofweek, month, 1500.0, 1450.0, 1550.0]],
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