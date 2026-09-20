import pandas as pd
import numpy as np

df = pd.read_csv("/home/claude/project/zomato_raw.csv")
orig_n = len(df)
log = [f"Starting rows: {orig_n}"]

# 1. Strip whitespace on all string columns
str_cols = [c for c in df.columns if df[c].dtype.kind not in 'if']
for c in str_cols:
    df[c] = df[c].astype("string").str.strip()

# 2. Parse dates & times
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format="%d-%m-%Y", errors='coerce')

def parse_time_flexible(series):
    """Handles both 'HH:MM' strings and Excel fractional-day floats (e.g. 0.4583 = 11:00)."""
    s = series.astype("string")
    std_parsed = pd.to_datetime(s, format="%H:%M", errors='coerce')
    # identify values that look like decimal fractions (Excel serial time export)
    is_fraction = s.str.match(r'^0?\.\d+$|^1$', na=False)
    frac_vals = pd.to_numeric(s.where(is_fraction), errors='coerce')
    minutes_total = (frac_vals * 24 * 60).round()
    hh = (minutes_total // 60).astype('Int64')
    mm = (minutes_total % 60).astype('Int64')
    frac_parsed = pd.to_datetime(
        hh.astype('string').str.zfill(2) + ':' + mm.astype('string').str.zfill(2),
        format="%H:%M", errors='coerce'
    )
    return std_parsed.fillna(frac_parsed)

df['Time_Orderd_parsed'] = parse_time_flexible(df['Time_Orderd'])
df['Time_Order_picked_parsed'] = parse_time_flexible(df['Time_Order_picked'])

# 3. Drop rows with missing critical fields for our analysis (date, order time, pickup time, target)
before = len(df)
df = df.dropna(subset=['Order_Date', 'Time_Orderd_parsed', 'Time_Order_picked_parsed', 'Time_taken (min)'])
log.append(f"Dropped {before - len(df)} rows missing critical timestamp/target fields -> {len(df)} remain")

# 4. Handle categorical nulls: impute with mode (documented assumption, not silently dropped)
for c in ['Weather_conditions', 'Road_traffic_density', 'Festival', 'City']:
    n_null = df[c].isna().sum()
    mode_val = df[c].mode(dropna=True)[0]
    df[c] = df[c].fillna(mode_val)
    log.append(f"Imputed {n_null} nulls in '{c}' with mode value '{mode_val}'")

# 5. Numeric nulls: median impute (age, ratings, multiple_deliveries)
for c in ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']:
    n_null = df[c].isna().sum()
    med = df[c].median()
    df[c] = df[c].fillna(med)
    log.append(f"Imputed {n_null} nulls in '{c}' with median value {med}")

# 6. Sanity bounds: ratings should be 1-5, age should be reasonable (18-60 for gig workers)
before = len(df)
df = df[(df['Delivery_person_Ratings'].between(1, 5)) & (df['Delivery_person_Age'].between(15, 65))]
log.append(f"Dropped {before - len(df)} rows with out-of-range age/rating -> {len(df)} remain")

# 7. Sanity bound on target: Time_taken should be positive and not absurd (cap at 120 min as plausible max)
before = len(df)
df = df[(df['Time_taken (min)'] > 0) & (df['Time_taken (min)'] <= 120)]
log.append(f"Dropped {before - len(df)} rows with implausible Time_taken -> {len(df)} remain")

# 8. Deduplicate on ID
before = len(df)
df = df.drop_duplicates(subset=['ID'])
log.append(f"Dropped {before - len(df)} duplicate IDs -> {len(df)} remain")

log.append(f"\nFinal cleaned shape: {df.shape}")
log.append(f"Retention rate: {len(df)/orig_n*100:.1f}% of original {orig_n} rows")

df.to_csv("/home/claude/project/artifacts/zomato_cleaned.csv", index=False)

with open("/home/claude/project/logs/phase2_log.txt", "w") as f:
    f.write("\n".join(log))

print("\n".join(log))
