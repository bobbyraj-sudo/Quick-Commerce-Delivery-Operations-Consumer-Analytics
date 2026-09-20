import pandas as pd
import numpy as np

df = pd.read_csv("/home/claude/project/zomato_raw.csv")

log = []
log.append(f"Raw shape: {df.shape}")
log.append(f"Columns: {list(df.columns)}")
log.append("\nDtypes:\n" + str(df.dtypes))
log.append("\nNull counts (raw):\n" + str(df.isnull().sum()))

# Check for hidden nulls encoded as strings
for col in df.columns:
    if df[col].dtype.kind not in 'if':  # skip pure numeric
        vals = df[col].map(lambda x: str(x).strip()).unique()
        weird = [v for v in vals if v.upper() in ('NAN', 'NA', 'NULL', '')]
        if weird:
            log.append(f"Hidden-null-like tokens in '{col}': {weird}")

log.append(f"\nDuplicate IDs: {df['ID'].duplicated().sum()}")
log.append(f"\nUnique cities: {df['City'].unique()}")
log.append(f"Unique weather values: {df['Weather_conditions'].unique()}")
log.append(f"Unique traffic values: {df['Road_traffic_density'].unique()}")
log.append(f"Unique vehicle types: {df['Type_of_vehicle'].unique()}")
log.append(f"Unique order types: {df['Type_of_order'].unique()}")
log.append(f"Festival values: {df['Festival'].unique()}")

with open("/home/claude/project/logs/phase1_log.txt", "w") as f:
    f.write("\n".join(log))

print("\n".join(log))
