import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
# TODO: update path & column names based on your dataset
df = pd.read_csv("data/matches.csv")

# Example columns (edit to match your data!)
feature_cols = [
    "home_shots",
    "away_shots",
    "home_possession",
    "away_possession",
    "home_odds",
    "away_odds",
]
target_col = "home_result"  # 1 = home win, 0 = not win (draw or away win)

X = df[feature_cols]
y = df[target_col]

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Match winner prediction accuracy: {acc:.3f}\n")
print(classification_report(y_test, y_pred))

# 6. Example: show first 5 predictions
results = X_test.copy()
results["true_label"] = y_test.values
results["predicted_label"] = y_pred
print("Sample predictions:")
print(results.head())