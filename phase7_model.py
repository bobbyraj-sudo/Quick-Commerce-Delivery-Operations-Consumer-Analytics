import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("/home/claude/project/artifacts/zomato_features.csv")
log = []

features_num = ['distance_km','multiple_deliveries','Delivery_person_Age',
                'Delivery_person_Ratings','Vehicle_condition','prep_lag_min']
features_cat = ['Road_traffic_density','Weather_conditions','Festival','City',
                 'Type_of_vehicle','day_part']
target = 'Time_taken (min)'

X = df[features_num + features_cat]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat)
], remainder='passthrough')

# --- Baseline: Linear Regression ---
lin_pipe = Pipeline([('pre', pre), ('model', LinearRegression())])
lin_pipe.fit(X_train, y_train)
pred_lin = lin_pipe.predict(X_test)
log.append(f"Linear Regression -> R2: {r2_score(y_test, pred_lin):.3f}, "
           f"MAE: {mean_absolute_error(y_test, pred_lin):.2f} min")

# --- Random Forest ---
rf_pipe = Pipeline([('pre', pre), ('model', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))])
rf_pipe.fit(X_train, y_train)
pred_rf = rf_pipe.predict(X_test)
log.append(f"Random Forest      -> R2: {r2_score(y_test, pred_rf):.3f}, "
           f"MAE: {mean_absolute_error(y_test, pred_rf):.2f} min")

# --- Feature importance from RF ---
ohe_cols = rf_pipe.named_steps['pre'].named_transformers_['cat'].get_feature_names_out(features_cat)
all_cols = list(ohe_cols) + features_num
importances = rf_pipe.named_steps['model'].feature_importances_
imp_series = pd.Series(importances, index=all_cols).sort_values(ascending=False).head(15)
log.append("\nTop 15 feature importances (Random Forest):\n" + str(imp_series))

# --- High-risk order classifier framing: flag predicted breach ---
df_test = X_test.copy()
df_test['actual'] = y_test.values
df_test['predicted'] = pred_rf
df_test['residual'] = df_test['actual'] - df_test['predicted']
worst = df_test.reindex(df_test['residual'].abs().sort_values(ascending=False).index).head(5)
log.append(f"\nModel error spread: mean abs residual={df_test['residual'].abs().mean():.2f} min, "
           f"90th pct abs residual={df_test['residual'].abs().quantile(0.9):.2f} min")

with open("/home/claude/project/logs/phase7_log.txt", "w") as f:
    f.write("\n".join(log))
print("\n".join(log))
