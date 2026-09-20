import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("/home/claude/project/artifacts/zomato_features.csv")
log = []

# 1. ANOVA: does traffic density group significantly affect Time_taken?
groups = [g['Time_taken (min)'].values for _, g in df.groupby('Road_traffic_density')]
f_stat, p_val = stats.f_oneway(*groups)
log.append(f"ANOVA (Time_taken ~ Road_traffic_density): F={f_stat:.2f}, p={p_val:.2e}")
log.append("  -> " + ("Statistically significant (p<0.05): traffic density genuinely drives delivery time."
                       if p_val < 0.05 else "Not significant."))

# 2. Welch's t-test: Festival vs non-Festival delivery times (unequal variance assumed)
fest_yes = df[df['Festival'] == 'Yes']['Time_taken (min)']
fest_no = df[df['Festival'] == 'No']['Time_taken (min)']
t_stat, p_val2 = stats.ttest_ind(fest_yes, fest_no, equal_var=False)
log.append(f"\nWelch's t-test (Festival vs Non-Festival Time_taken): t={t_stat:.2f}, p={p_val2:.2e}")
log.append(f"  Festival mean={fest_yes.mean():.1f} min (n={len(fest_yes)}), "
           f"Non-Festival mean={fest_no.mean():.1f} min (n={len(fest_no)})")
log.append("  -> " + ("Statistically significant difference." if p_val2 < 0.05 else "Not significant."))

# 3. Chi-square: Festival vs SLA breach (categorical association)
contingency = pd.crosstab(df['Festival'], df['sla_breach'])
chi2, p_val3, dof, expected = stats.chi2_contingency(contingency)
log.append(f"\nChi-square test (Festival x SLA_breach): chi2={chi2:.2f}, dof={dof}, p={p_val3:.2e}")
log.append(str(contingency))
log.append("  -> " + ("Significant association." if p_val3 < 0.05 else "No significant association."))

# 4. Pearson correlation significance: distance_km vs Time_taken
r, p_val4 = stats.pearsonr(df['distance_km'], df['Time_taken (min)'])
log.append(f"\nPearson correlation (distance_km vs Time_taken): r={r:.3f}, p={p_val4:.2e}")

# 5. Effect size check: is the 'prep_lag_min ~0 correlation' a real null finding or noise?
r2, p_val5 = stats.pearsonr(df['prep_lag_min'].fillna(df['prep_lag_min'].median()), df['Time_taken (min)'])
log.append(f"\nPearson correlation (prep_lag_min vs Time_taken): r={r2:.3f}, p={p_val5:.2e}")
log.append("  -> " + ("Despite large n, effect is negligible: kitchen-side prep delay is NOT "
                       "a meaningful driver of total delivery time in this dataset." if abs(r2) < 0.05
                       else "Some relationship detected."))

with open("/home/claude/project/logs/phase5_log.txt", "w") as f:
    f.write("\n".join(log))
print("\n".join(log))
