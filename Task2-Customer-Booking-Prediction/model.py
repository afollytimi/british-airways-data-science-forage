import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

df = pd.read_csv("customer_booking.csv", encoding="latin1")

# ---------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------
df["is_weekend_flight"] = df["flight_day"].isin(["Sat", "Sun"]).astype(int)

day_map = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
df["flight_day_num"] = df["flight_day"].map(day_map)

# Route/origin have high cardinality (799 routes, 104 origins) -> frequency-encode
route_freq = df["route"].value_counts(normalize=True)
df["route_freq"] = df["route"].map(route_freq)

origin_freq = df["booking_origin"].value_counts(normalize=True)
df["booking_origin_freq"] = df["booking_origin"].map(origin_freq)

# Purchase-lead / stay-length buckets & extras engagement
df["purchase_lead_log"] = np.log1p(df["purchase_lead"])
df["length_of_stay_log"] = np.log1p(df["length_of_stay"])
df["total_extras_wanted"] = (df["wants_extra_baggage"] + df["wants_preferred_seat"]
                              + df["wants_in_flight_meals"])
df["is_short_lead"] = (df["purchase_lead"] <= 7).astype(int)  # booked within a week of travel
df["is_business_hour_flight"] = df["flight_hour"].between(6, 9).astype(int)

cat_cols = ["sales_channel", "trip_type"]
df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

feature_cols = [
    "num_passengers", "purchase_lead", "purchase_lead_log", "length_of_stay",
    "length_of_stay_log", "flight_hour", "flight_day_num", "is_weekend_flight",
    "is_business_hour_flight", "route_freq", "booking_origin_freq",
    "wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals",
    "total_extras_wanted", "is_short_lead", "flight_duration",
] + [c for c in df_model.columns if c.startswith("sales_channel_") or c.startswith("trip_type_")]

X = df_model[feature_cols]
y = df_model["booking_complete"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

model = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=5,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)

# 5-fold stratified cross-validation on the training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

cv_summary = {m: (cv_results[f"test_{m}"].mean(), cv_results[f"test_{m}"].std()) for m in scoring}

# Fit on full training set, evaluate on held-out test set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

test_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# ---------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5.5))
top = importances.head(12).iloc[::-1]
colors = ["#C8102E" if v == top.max() else "#0F1B3C" for v in top.values]
ax.barh(top.index, top.values, color=colors)
ax.set_xlabel("Relative importance (Gini)")
ax.set_title("What drives a completed booking? — Top 12 features", fontsize=12, fontweight="bold")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
plt.close()

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(4.2, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["No booking", "Booking"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No booking", "Booking"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Test Set)", fontsize=11, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()

print("=== 5-fold CV (train set) ===")
for m, (mean, std) in cv_summary.items():
    print(f"{m:10s}: {mean:.3f} +/- {std:.3f}")

print("\n=== Held-out test set ===")
for m, v in test_metrics.items():
    print(f"{m:10s}: {v:.3f}")

print("\nConfusion matrix:\n", cm)
print("\n", report)

print("\n=== Top 12 feature importances ===")
print(importances.head(12))

# Save numeric outputs for the slide build step
import json
with open("model_results.json", "w") as f:
    json.dump({
        "cv_summary": {k: [float(v[0]), float(v[1])] for k, v in cv_summary.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "confusion_matrix": cm.tolist(),
        "top_features": importances.head(8).round(4).to_dict(),
        "n_rows": int(len(df)),
        "positive_rate": float(y.mean()),
    }, f, indent=2)
print("\nSaved model_results.json, feature_importance.png, confusion_matrix.png")
