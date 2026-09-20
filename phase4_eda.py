import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv("/home/claude/project/artifacts/zomato_features.csv")
CHARTS = "/home/claude/project/charts"
log = [f"EDA on {len(df)} cleaned & feature-engineered rows"]

# 1. Distribution of delivery time
plt.figure(figsize=(7,4))
sns.histplot(df['Time_taken (min)'], bins=40, kde=True, color="#C00000")
plt.axvline(df['Time_taken (min)'].mean(), color='black', linestyle='--', label=f"Mean={df['Time_taken (min)'].mean():.1f}")
plt.axvline(df['Time_taken (min)'].median(), color='blue', linestyle='--', label=f"Median={df['Time_taken (min)'].median():.1f}")
plt.title("Distribution of Delivery Time (min)")
plt.xlabel("Time taken (min)"); plt.legend()
plt.tight_layout(); plt.savefig(f"{CHARTS}/01_time_distribution.png", dpi=130); plt.close()
skew = df['Time_taken (min)'].skew()
log.append(f"Time_taken skewness: {skew:.2f} (right-skewed -> median more representative than mean)")

# 2. Traffic density vs delivery time
plt.figure(figsize=(7,4))
order = ['Low','Medium','High','Jam']
sns.boxplot(data=df, x='Road_traffic_density', y='Time_taken (min)', order=order, palette="Reds")
plt.title("Delivery Time by Road Traffic Density")
plt.tight_layout(); plt.savefig(f"{CHARTS}/02_traffic_vs_time.png", dpi=130); plt.close()
log.append("\nMedian delivery time by traffic density:\n" + str(df.groupby('Road_traffic_density')['Time_taken (min)'].median().reindex(order)))

# 3. Weather vs delivery time
plt.figure(figsize=(8,4))
sns.boxplot(data=df, x='Weather_conditions', y='Time_taken (min)', palette="Blues")
plt.title("Delivery Time by Weather Condition")
plt.xticks(rotation=20)
plt.tight_layout(); plt.savefig(f"{CHARTS}/03_weather_vs_time.png", dpi=130); plt.close()
log.append("\nMedian delivery time by weather:\n" + str(df.groupby('Weather_conditions')['Time_taken (min)'].median().sort_values(ascending=False)))

# 4. Festival impact
plt.figure(figsize=(5,4))
sns.boxplot(data=df, x='Festival', y='Time_taken (min)', palette="Purples")
plt.title("Delivery Time: Festival vs Non-Festival")
plt.tight_layout(); plt.savefig(f"{CHARTS}/04_festival_vs_time.png", dpi=130); plt.close()
fest_median = df.groupby('Festival')['Time_taken (min)'].median()
log.append(f"\nMedian delivery time -> Festival: {fest_median.get('Yes', float('nan')):.1f} min, "
           f"Non-Festival: {fest_median.get('No', float('nan')):.1f} min")

# 5. SLA breach rate by hour-of-day and city (heatmap = 'control tower' view)
pivot = df.pivot_table(index='City', columns='order_hour', values='sla_breach', aggfunc='mean') * 100
plt.figure(figsize=(12,3.5))
sns.heatmap(pivot, cmap="Reds", annot=False, cbar_kws={'label': 'SLA breach %'})
plt.title("SLA Breach Rate (%) by City x Hour-of-Day")
plt.xlabel("Order Hour"); plt.ylabel("City Type")
plt.tight_layout(); plt.savefig(f"{CHARTS}/05_breach_heatmap.png", dpi=130); plt.close()

worst_cells = pivot.stack().sort_values(ascending=False).head(5)
log.append(f"\nTop 5 worst City x Hour cells by SLA breach rate:\n{worst_cells}")

# 6. Distance vs delivery time by vehicle type
plt.figure(figsize=(7,5))
sns.scatterplot(data=df.sample(min(5000,len(df)), random_state=1), x='distance_km', y='Time_taken (min)',
                 hue='Type_of_vehicle', alpha=0.4, s=15)
plt.title("Distance vs Delivery Time by Vehicle Type (sampled)")
plt.tight_layout(); plt.savefig(f"{CHARTS}/06_distance_vehicle.png", dpi=130); plt.close()

# 7. Multiple deliveries impact
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='multiple_deliveries', y='Time_taken (min)', palette="Greens")
plt.title("Delivery Time by Number of Concurrent Deliveries")
plt.tight_layout(); plt.savefig(f"{CHARTS}/07_multideliveries_vs_time.png", dpi=130); plt.close()
log.append("\nMedian delivery time by multiple_deliveries count:\n" + str(df.groupby('multiple_deliveries')['Time_taken (min)'].median()))

# 8. Correlation matrix (numeric features)
num_cols = ['Delivery_person_Age','Delivery_person_Ratings','distance_km','prep_lag_min',
            'multiple_deliveries','Vehicle_condition','Time_taken (min)']
corr = df[num_cols].corr()
plt.figure(figsize=(7,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix — Numeric Features")
plt.tight_layout(); plt.savefig(f"{CHARTS}/08_correlation_matrix.png", dpi=130); plt.close()
log.append("\nCorrelation with Time_taken (min):\n" + str(corr['Time_taken (min)'].sort_values(ascending=False)))

with open("/home/claude/project/logs/phase4_log.txt", "w") as f:
    f.write("\n".join(log))
print("\n".join(log))
print("\nCharts saved:", sorted([f for f in __import__('os').listdir(CHARTS)]))
