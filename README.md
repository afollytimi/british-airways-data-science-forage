# British Airways Data Science Job Simulation (Forage)

My work from the [Forage](https://www.theforage.com/) British Airways Data Science job simulation, covering both tasks.

## Task 1 — Modeling Lounge Eligibility at Heathrow Terminal 3

**Goal:** Build a reusable lookup table British Airways can use to estimate lounge eligibility percentages (Concorde Room / First Lounge / Club Lounge) across future flight schedules, without needing exact flight or aircraft data.

**Approach:** Grouped Terminal 3 traffic by route type/region and time of day (8 categories — UK regional, short-haul Europe AM/PM, long-haul North America AM/PM, Middle East & Asia, Africa, Caribbean & leisure) and assigned reasoned Tier 1/2/3 eligibility percentages to each group based on known differences in BA's cabin mix and route demographics.

**Deliverable:** [`Task1-Lounge-Eligibility/Lounge_Eligibility_Lookup_and_Justification.xlsx`](./Task1-Lounge-Eligibility/Lounge_Eligibility_Lookup_and_Justification.xlsx) — a two-sheet workbook with the lookup table and a written justification (grouping rationale, assumptions, and scalability to future schedules).

## Task 2 — Predicting Customer Buying Behaviour

**Goal:** Prepare a customer booking dataset, train a machine learning model to predict whether a booking will be completed, evaluate it, and summarize findings for a manager in a single slide.

**Approach:**
- Cleaned and feature-engineered the 50,000-row `customer_booking.csv` dataset (log-scaled purchase lead time and length of stay, route/booking-origin frequency encoding, weekend and business-hour flags, total extras requested).
- Trained a class-balanced **Random Forest classifier** (300 trees) to predict `booking_complete`.
- Validated with 5-fold stratified cross-validation and a held-out test set; visualized feature importance and the confusion matrix.

**Results (held-out test set, n=10,000):**

| Metric | Score |
|---|---|
| ROC-AUC | 0.766 |
| Accuracy | 73.3% |
| Recall (true bookers) | 63.4% |
| Precision (true bookers) | 30.8% |

**Key finding:** Booking origin (country) is by far the strongest predictor — roughly 30% of the model's predictive power, nearly 3x the next-strongest driver (length of stay). Trip-planning behaviour (purchase lead time, length of stay) and route are the next strongest signals.

**Files:**
- [`Task2-Customer-Booking-Prediction/model.py`](./Task2-Customer-Booking-Prediction/model.py) — full pipeline: feature engineering, training, cross-validation, evaluation, plots
- [`Task2-Customer-Booking-Prediction/model_results.json`](./Task2-Customer-Booking-Prediction/model_results.json) — CV and test-set metrics, confusion matrix, top feature importances
- [`Task2-Customer-Booking-Prediction/feature_importance.png`](./Task2-Customer-Booking-Prediction/feature_importance.png) — top 12 feature importances
- [`Task2-Customer-Booking-Prediction/confusion_matrix.png`](./Task2-Customer-Booking-Prediction/confusion_matrix.png) — test-set confusion matrix
- [`Task2-Customer-Booking-Prediction/Customer_Booking_Prediction_Summary.pptx`](./Task2-Customer-Booking-Prediction/Customer_Booking_Prediction_Summary.pptx) — one-slide manager summary

> Note: the raw `customer_booking.csv` dataset is excluded from this repo, as Forage marks it "for Forage simulation use only."

## Stack

Python (pandas, scikit-learn, matplotlib), openpyxl, pptxgenjs.
