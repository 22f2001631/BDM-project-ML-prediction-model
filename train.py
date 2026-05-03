import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)

# LOAD DATA
df = pd.read_csv("./data/records.csv")

# CLEAN NUMERIC
numeric_cols = [
    "Total Revenue", "Total Expense", "Profit"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# ENCODING
route_encoder = LabelEncoder()
day_encoder = LabelEncoder()
phase_encoder = LabelEncoder()

df["Route_enc"] = route_encoder.fit_transform(df["Route ID"])
df["Day_enc"] = day_encoder.fit_transform(df["Day"])

# CREATE PHASE
df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
df["Month"] = df["Date"].dt.month

def get_phase(m):
    if m == 4:
        return "Phase1"
    elif m in [5, 6]:
        return "Phase2"
    else:
        return "Phase3"

df["Phase"] = df["Month"].apply(get_phase)
df["Phase_enc"] = phase_encoder.fit_transform(df["Phase"])

# FEATURES
X = df[["Route_enc", "Day_enc", "Phase_enc"]]

# TARGET (MULTI OUTPUT)
y = df[["Total Revenue", "Total Expense", "Profit"]]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# EVAL
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# SAVE
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump((route_encoder, day_encoder, phase_encoder), f)

print("Model saved successfully")