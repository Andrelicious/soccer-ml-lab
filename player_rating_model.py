import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load data
# TODO: update path & column names based on your dataset
df = pd.read_csv("data/players.csv")

# Example columns (edit to match your data!)
feature_cols = [
    "minutes_played",
    "shots",
    "key_passes",
    "tackles",
    "interceptions",
    "successful_dribbles",
]
target_col = "rating"  # e.g. match rating from 0–10 or 0–100

X = df[feature_cols]
y = df[target_col]

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train regression model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Player rating model MAE: {mae:.3f}")
print(f"Player rating model R²:  {r2:.3f}")

# 5. Sample predictions
results = X_test.copy()
results["true_rating"] = y_test.values
results["predicted_rating"] = y_pred
print("\nSample player ratings:")
print(results.head())