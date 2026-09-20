import pandas as pd
import numpy as np

df = pd.read_csv("/home/claude/project/artifacts/zomato_cleaned.csv", parse_dates=['Order_Date'])
log = [f"Input rows: {len(df)}"]

# --- Haversine distance between restaurant and delivery location ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine(
    df['Restaurant_latitude'], df['Restaurant_longitude'],
    df['Delivery_location_latitude'], df['Delivery_location_longitude']
)
log.append(f"Distance stats (km): min={df['distance_km'].min():.2f}, "
           f"median={df['distance_km'].median():.2f}, max={df['distance_km'].max():.2f}")

# Some lat/long pairs in this dataset are known to contain placeholder/erroneous coordinates
# (near 0,0 or absurd distances) -- cap at 99th percentile rather than silently trusting it
p99 = df['distance_km'].quantile(0.99)
before = len(df)
df = df[df['distance_km'] <= p99]
log.append(f"Capped distance outliers above 99th percentile ({p99:.1f} km) -> dropped {before-len(df)} rows")

# --- Order-to-pickup lag (kitchen-side delay) ---
t_order = pd.to_datetime(df['Time_Orderd_parsed']).dt.hour * 60 + pd.to_datetime(df['Time_Orderd_parsed']).dt.minute
t_pick = pd.to_datetime(df['Time_Order_picked_parsed']).dt.hour * 60 + pd.to_datetime(df['Time_Order_picked_parsed']).dt.minute
lag = t_pick - t_order
lag = np.where(lag < 0, lag + 24*60, lag)  # handle midnight rollover
df['prep_lag_min'] = lag
log.append(f"Prep-lag stats (min): median={np.median(lag):.1f}, 95th pct={np.percentile(lag,95):.1f}")

# --- Time features ---
df['order_hour'] = pd.to_datetime(df['Time_Orderd_parsed']).dt.hour
df['order_dow'] = df['Order_Date'].dt.day_name()

def hour_bucket(h):
    if 6 <= h < 11: return "Morning (6-11)"
    if 11 <= h < 15: return "Lunch (11-15)"
    if 15 <= h < 19: return "Afternoon (15-19)"
    if 19 <= h < 23: return "Dinner (19-23)"
    return "Late Night (23-6)"
df['day_part'] = df['order_hour'].apply(hour_bucket)

# --- SLA breach definition (explicit, documented assumption) ---
# No promised-SLA field exists in this dataset -> define breach as exceeding the
# 75th percentile of Time_taken, computed PER CITY TYPE (since Metropolitan/Urban/
# Semi-Urban have structurally different baseline delivery times).
city_p75 = df.groupby('City')['Time_taken (min)'].transform(lambda x: x.quantile(0.75))
df['sla_threshold_min'] = city_p75
df['sla_breach'] = (df['Time_taken (min)'] > df['sla_threshold_min']).astype(int)

log.append("\nSLA threshold by city (75th percentile of Time_taken):")
log.append(str(df.groupby('City')['Time_taken (min)'].quantile(0.75)))
log.append(f"\nOverall breach rate: {df['sla_breach'].mean()*100:.1f}%")

df.to_csv("/home/claude/project/artifacts/zomato_features.csv", index=False)

with open("/home/claude/project/logs/phase3_log.txt", "w") as f:
    f.write("\n".join(log))

print("\n".join(log))
