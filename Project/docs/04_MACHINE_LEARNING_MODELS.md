# 04 · The Machine Learning Models

> Files: `modeling/*.py`, `data_pipeline/feature_builder.py`, `run_model_training.py`, `tools/prediction_tools.py`

The project trains **three** models. Each one answers a question the desk actually has, and none of them decides whether a spot may air. That is done by deterministic rule checks.

| Model | Question it answers | Type | Algorithm | Held-out score | Minimum |
|---|---|---|---|---|---|
| **Amendment** | Will this campaign's copy need changes before it can be cleared? | Binary classification | Logistic Regression | **ROC AUC 0.8571** | 0.70 |
| **Exposure** | How many viewers under 18 will this programme have (thousands)? | Regression | HistGradientBoosting, Poisson loss | **R² 0.9213** | 0.60 |
| **Issue** | Which of 8 copy issues does this clearance note describe? | Multi-class text (NLP) | TF-IDF + Logistic Regression | **Macro F1 0.8624** | 0.75 |

## 1. The shared foundation: a Pipeline saved with the model

```
build_feature_frame()                 build_model_pipeline()
select config's columns,      ┌──────────────── sklearn Pipeline (saved as one .pkl) ────────────────┐
numbers → float, categories → │ ColumnTransformer                                                    │
text, any gap → NaN       ──▶ │  numbers:    SimpleImputer(median, add_indicator) → StandardScaler    │──▶ estimator
                              │  categories: SimpleImputer("Not recorded") → OneHotEncoder(ignore)   │
                              └──────────────────────────────────────────────────────────────────────┘
```

**Why a Pipeline?**
1. **No data leakage.** The median, the scaling and the category list are learned **only from the training rows**, inside `fit`.
2. **Identical preparation at scoring time.** The live campaigns go through exactly the same steps, because the steps are saved inside the model file.
3. **Robustness.** A missing value is filled (and `add_indicator` tells the model "this was missing", which itself can be a signal). A category never seen in training (`handle_unknown="ignore"`) is scored without an error.

**Feature schemas** (`output/models/*_features.json`) record the exact column order. Live rows are built in that order, and the test suite checks the model's `feature_names_in_` match.

**Inputs come only from config** (`AMENDMENT_NUMERIC_FEATURES`, etc.). Identifiers (`submission_id`, `clock_number`, `batch_code` …) and the answer are never inputs. They would let the model memorise rows instead of learning patterns.

---

## 2. The amendment model (classification)

**Data:** 29,697 cleaned training submissions; scored on 8,224 held-out ones. About **29%** needed changes.

**Inputs:** copy length, supers, legal text seconds, child-appeal elements, claims, booking shape (spots, channels, share before 21:00, share in live sport), days to first transmission, the advertiser's and agency's history, plus categorical flags (health/price/environmental/comparative claims, offer terms shown, bonus/free bet, APR shown, celebrity, performer age band, category, sub-category …).

**Why Logistic Regression?** We compared it with gradient boosting on the same held-out data:

| | Logistic Regression | Gradient Boosting |
|---|---|---|
| Held-out ROC AUC | **0.8571** | 0.8460 |
| Train ROC AUC | 0.8639 | 0.9021 |
| Generalisation gap | **0.007** (barely memorises) | 0.056 |
| Mean predicted vs actual share | 0.2907 vs 0.2917 (**well calibrated**) | 0.2910 |

It ranks better, generalises better, and gives **calibrated probabilities**. That matters because the score is banded at fixed thresholds: **HIGH ≥ 0.60, MEDIUM ≥ 0.30, LOW** below. Its coefficients are also **explainable**: each flag pushes the score up or down by a readable amount.

**Sanity checks it passes** (the business logic the model must have learned):
- The same copies with a **health claim** score higher on average than without one (0.52 vs 0.23).
- An offer with its **terms left off screen** scores higher than with terms shown (0.70 vs 0.35).
- The live queue's scores spread across all three bands (standard deviation > 0.05).

**Evaluation metrics and why:**
- **ROC AUC** (main metric): the probability that a random amended copy scores higher than a random clean one. It measures **ranking**, which is exactly what a "read this first" score is for, and it isn't distorted by the 29/71 class imbalance.
- **Accuracy 0.81, precision 0.72, recall 0.56, F1 0.63** (at a 0.5 threshold) are reported for completeness. Precision is the share of predicted-amendments that really needed changes; recall is the share of real amendments caught.
- **Generalisation gap** = train AUC − held-out AUC. Small means it isn't overfitting.
- **Mean predicted probability vs positive share** checks calibration.

> **Important:** the score says how likely the copy is to *need changes*. It does **not** say whether a spot breaks a code. Every agent prompt says so.

---

## 3. The exposure model (regression)

**Data:** 29,817 training airings; scored on 6,770 airings from ten other weeks.

**Inputs (known before the programme airs):** channel, broadcast day, genre, lead-in genre, live, live sport, made for children, premiere, episode number, school holidays, day type, forecast index, and the **slot**: `start_minute` on the broadcast clock (e.g. 25:30 → 1530) and `duration_minutes`.
`audience_all` (the total audience) is **never** an input: it is only measured *after* airing, so it can't be known for next week's schedule. Using it would be **target leakage**.

**Why the broadcast-clock minute?** On the ordinary clock, 01:30 (90) would look "earlier" than 06:00 (360). On the broadcast clock it is 25:30 (1530), correctly **later** than the evening. The model can then learn that the small hours have few young viewers.

**Why HistGradientBoosting with Poisson loss?**

| | Squared-error GBM | **Poisson GBM** |
|---|---|---|
| Held-out R² | 0.9183 | **0.9213** |
| Lowest forecast | **−4.18** (a negative audience!) | **0.415** (always positive) |

- Audiences are **non-negative and heavily skewed** (median 12.2 thousand, max 1,563.7). That's the shape Poisson regression is designed for.
- Poisson loss predicts through `exp(...)`, so **every forecast is above zero**. That guarantees a campaign with a breaching spot always has exposure > 0.
- Boosted trees capture **interactions** (e.g. "children's genre × morning slot × school holidays") without hand-made features.

**Sanity checks it passes:**
- **School holidays** forecast more young viewers than term time (47.5 vs 45.6).
- **19:00–20:00** forecasts far more than **25:00–26:00** (93.1 vs 4.4).
- An index **50 points higher** forecasts more (65.9 vs 46.2).

**Metrics:**
- **R²** (main): the share of audience variation explained. 0.92 means the forecasts track reality closely.
- **MAE 9.46 thousand** vs **51.35** for always guessing the training average. That's the *baseline* comparison; the model is about 5× better.
- **Lowest prediction** proves positivity. The **generalisation gap** (0.029) is small.

**How it's used:** every booking-week programme is forecast once at start-up. A campaign's **exposure** = the sum, over its breaching spots, of the forecast audience of the programme the spot is in (for an end break, the larger of the two). The queue is ranked by this.

---

## 4. The issue model (NLP text classification)

**Data:** 33,045 cleaned notes, split once **80/20, stratified by issue, `random_state=42`**: 26,436 to train, 6,609 held out.

**Pipeline:** `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` → `LogisticRegression(class_weight="balanced")`.

**Why this combination?**
- Notes are short and issue-specific vocabulary is the signal ("free bet", "APR", "recyclable", "clinically proven"). **TF-IDF** weights words that are frequent in a note but rare overall. **Bigrams** keep phrases like "no evidence" together. `sublinear_tf` dampens repeated words; `min_df=2` drops one-off typos.
- **Logistic regression** gives a **probability for every issue**, which the weighted vote and its tie-break need.
- **`class_weight="balanced"`** stops common issues (Health claim 22.6%) from drowning out rare ones (Environmental claim 4.2%). The main metric is **macro** F1.
- It's fast and transparent. A transformer would be heavier and slower, and isn't needed to clear 0.75 comfortably.

**Metrics:**
- **Macro F1 0.8624** (main): F1 computed per issue, then averaged equally. The rarest issue counts as much as the commonest. That's the point, because rare issues are the ones a busy desk misreads.
- Accuracy 0.886, weighted F1 0.889, and **per-class F1** for every issue (lowest: Environmental claim 0.73; highest: Health claim 0.92).
- Every issue is predicted for some held-out note, and each issue's notes are most often read as that issue.

**How it's used (the weighted vote, `build_issue_facts`):**
1. Notes under 6 words are set aside (as in training).
2. Each readable note votes for its most probable issue with weight **2 for a Legal review, 1 otherwise**.
3. The issue with the most weighted votes wins. A **tie** goes to the issue with the greatest summed (weighted) probability, never to whichever note came first.
4. **Weighted agreement** = winner's votes ÷ total votes (e.g. 3 of 4 = 0.75). Below about 0.5, the notes disagree.

---

## 5. What `run_model_training.py` records
`output/model_metrics.json` holds, per model: the algorithm, training and held-out row counts, all scores, the minimum required, `meets_minimum`, and the scikit-learn version (for reproducibility). A model below its minimum is **reported, not raised**, so the run still finishes.
