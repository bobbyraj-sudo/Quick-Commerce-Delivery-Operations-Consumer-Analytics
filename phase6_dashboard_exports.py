import pandas as pd

df = pd.read_csv("/home/claude/project/artifacts/zomato_features.csv")
OUT = "/home/claude/project/artifacts"

# 1. Zone(City) x Hour SLA breach table -> heatmap panel
zone_hour = (df.groupby(['City','order_hour'])
               .agg(orders=('ID','count'), breach_rate=('sla_breach','mean'),
                    avg_time=('Time_taken (min)','mean'))
               .reset_index())
zone_hour['breach_rate'] = (zone_hour['breach_rate']*100).round(1)
zone_hour.to_csv(f"{OUT}/dash_zone_hour_breach.csv", index=False)

# 2. Daily trend table -> trend line panel
daily = (df.groupby('Order_Date')
           .agg(orders=('ID','count'), avg_time=('Time_taken (min)','mean'),
                breach_rate=('sla_breach','mean'))
           .reset_index())
daily['breach_rate'] = (daily['breach_rate']*100).round(1)
daily.to_csv(f"{OUT}/dash_daily_trend.csv", index=False)

# 3. Traffic/Weather impact panel
tw = (df.groupby(['Road_traffic_density','Weather_conditions'])
        .agg(orders=('ID','count'), avg_time=('Time_taken (min)','mean'),
             breach_rate=('sla_breach','mean'))
        .reset_index())
tw['breach_rate'] = (tw['breach_rate']*100).round(1)
tw.to_csv(f"{OUT}/dash_traffic_weather.csv", index=False)

# 4. Festival surge summary card
fest = (df.groupby('Festival')
          .agg(orders=('ID','count'), avg_time=('Time_taken (min)','mean'),
               breach_rate=('sla_breach','mean'))
          .reset_index())
fest['breach_rate'] = (fest['breach_rate']*100).round(1)
fest.to_csv(f"{OUT}/dash_festival_summary.csv", index=False)

# 5. Rider-load (multiple_deliveries) staffing panel
load = (df.groupby('multiple_deliveries')
          .agg(orders=('ID','count'), avg_time=('Time_taken (min)','mean'),
               breach_rate=('sla_breach','mean'))
          .reset_index())
load['breach_rate'] = (load['breach_rate']*100).round(1)
load.to_csv(f"{OUT}/dash_rider_load.csv", index=False)

print("Dashboard export tables written:")
for f in ["dash_zone_hour_breach.csv","dash_daily_trend.csv","dash_traffic_weather.csv",
          "dash_festival_summary.csv","dash_rider_load.csv"]:
    d = pd.read_csv(f"{OUT}/{f}")
    print(f" - {f}: {d.shape[0]} rows x {d.shape[1]} cols")
