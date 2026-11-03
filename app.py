import datetime
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# Load Saved Models and Dataset for dynamic lag lookup
@st.cache_resource
def load_models_and_data():
  solar_model = joblib.load("solar_model.pkl")
  load_model = joblib.load("load_model.pkl")

  # Load dataset to extract dynamic historical load patterns for lag features
  df_load = pd.read_csv("household_data_60min_singleindex.csv")
  cols = [
      "utc_timestamp",
      "DE_KN_residential3_grid_import",
      "DE_KN_residential3_grid_export",
      "DE_KN_residential3_pv",
  ]
  df_load_sub = df_load[cols].copy()
  df_load_sub["utc_timestamp"] = pd.to_datetime(df_load_sub["utc_timestamp"])
  df_load_sub = df_load_sub.sort_values("utc_timestamp").reset_index(drop=True)
  df_load_sub.fillna(0, inplace=True)
  df_load_sub["total_load"] = (
      df_load_sub["DE_KN_residential3_grid_import"]
      + df_load_sub["DE_KN_residential3_pv"]
      - df_load_sub["DE_KN_residential3_grid_export"]
  )
  return solar_model, load_model, df_load_sub


solar_model, load_model, df_load_sub = load_models_and_data()

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
    "Select your **local date and time in Sri Lanka (GMT+5:30)** and your **Solar Panel Area**. The app fetches live weather including rain/precipitation, scales solar output based on your system size, and simulates energy management."
)

# Sidebar Inputs (Local Date, Time & Panel Area)
st.sidebar.header("📅 Simulation Parameters")
selected_date = st.sidebar.date_input(
    "Target Date", datetime.date.today()
)
selected_time = st.sidebar.time_input(
    "Target Local Time", datetime.time(14, 0)
)

# New User Input: Solar Panel Area in square meters (m²)
# Formula reference: Prated = A_module * eta_STC * 1000 W/m² (Assuming 20% efficiency -> 1 m² = 0.2 kWp)
module_area = st.sidebar.number_input(
    "Solar Panel Area (m²)",
    min_value=1.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
    help="Enter total rooftop module area. Assumes standard 20% module efficiency.",
)

local_datetime = datetime.datetime.combine(selected_date, selected_time)
sri_lanka_offset = datetime.timedelta(hours=5, minutes=30)
utc_datetime = local_datetime - sri_lanka_offset

local_hour = local_datetime.hour
month = local_datetime.month
dayofyear = local_datetime.timetuple().tm_yday
dayofweek = local_datetime.weekday()

st.sidebar.divider()
battery_soc = st.sidebar.slider(
    "Current Battery SoC (%)", min_value=0.0, max_value=100.0, value=50.0
)


# Fetch Real Weather & Precipitation from Open-Meteo
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

  # Fallback
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


temp, humidity, wind_speed, precip, direct_irr, diffuse_irr, sun_height, is_live = (
    fetch_open_meteo_weather(utc_datetime)
)

status_badge = (
    "🟢 Live Open-Meteo Data" if is_live else "🟡 Estimated / Fallback Data"
)
st.info(
    f"🇱🇰 **Local Time: {local_datetime.strftime('%Y-%m-%d %H:%M')} (Sri Lanka)** | Status: {status_badge} | "
    f"Temp: {temp:.1f}°C | Rain/Precipitation: {precip:.1f} mm/h | Wind: {wind_speed} m/s"
)

# 1. Solar Prediction for 1 kWp model, then scaled by user's panel area
# Assuming 20% module efficiency: 1 m² area produces 0.2 kWp capacity rated at STC (1000 W/m²)
module_efficiency = 0.20
rated_capacity_kwp = (module_area * module_efficiency)  # e.g., 10 m² = 2 kWp

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

# Base prediction for 1 kWp system
predicted_solar_1kwp = float(solar_model.predict(solar_input_df)[0])
predicted_solar_1kwp = max(0.0, predicted_solar_1kwp)

# Scale by user's system rated capacity (kWp)
predicted_solar = predicted_solar_1kwp * rated_capacity_kwp

# 2. Dynamic Load Prediction
matching_hours = df_load_sub[
    (df_load_sub["utc_timestamp"].dt.hour == local_hour)
    & (df_load_sub["utc_timestamp"].dt.dayofweek == dayofweek)
]
if len(matching_hours) > 0:
  avg_load = matching_hours["total_load"].mean()
  lag_24 = avg_load * 0.98
  lag_48 = avg_load * 1.02
  lag_168 = avg_load * 1.00
else:
  lag_24, lag_48, lag_168 = 1500.0, 1450.0, 1550.0

load_input_df = pd.DataFrame(
    [[local_hour, dayofweek, month, lag_24, lag_48, lag_168]],
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
predicted_load += (temp - 25.0) * 25.0
predicted_load = max(100.0, predicted_load)

net_load = predicted_load - predicted_solar

# Display Metrics
col1, col2, col3 = st.columns(3)
with col1:
  st.metric(
      label=f"☀️ Predicted Solar Generation ({module_area} m² / {rated_capacity_kwp:.1f} kWp)",
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

# Maintenance Alerting (incorporating precipitation/rain check)
st.divider()
st.subheader("🛠️ Condition-Based Maintenance (CBM) Alerting")
actual_observed_solar = predicted_solar * (
    0.75 if precip > 0 else 0.85
)  # Lower output during rain/clouds
performance_ratio = (
    (actual_observed_solar / predicted_solar) * 100 if predicted_solar > 0 else 100
)

if precip > 0:
  st.info(
      f"🌧️ **Rain Detected ({precip:.1f} mm/h)**: Solar reduction is due to active rainfall/cloud cover rather than a fault. System operating normally."
  )
elif performance_ratio < 85.0 and (direct_irr + diffuse_irr) > 300:
  st.warning(
      "⚠️ **[Maintenance Alert Triggered]**: Actual solar generation is sustained >15% below model expectations under high irradiance. **Recommended Action:** Inspect panels for soiling or shading."
  )
else:
  st.info(
      "✅ **System Status Healthy**: Solar PV generation matches model expectations within acceptable tolerances."
  )