# Rehabilitating Homeless: Dataset, Reproducibility Audit, and Extended Modeling Analyses

This repository contains the anonymized dataset, baseline reproduction code (**Deliverable B**), and extended modeling analyses (**Deliverable C**) for the paper *"Rehabilitating Homeless: Dataset and Key Insights"* (Bykova et al., AAAI-23 Social Impact Track).

The dataset is based on over 9 years of longitudinal case management and data collection by **"Nochlezhka"** ([https://homeless.ru/en/how_to_help/](https://homeless.ru/en/how_to_help/)), the largest non-governmental organization assisting homeless individuals in Russia.

---

## Repository Structure & Deliverables

| File | Deliverable / Role | Description |
| :--- | :--- | :--- |
| `nochlezhka_rel_2.ipynb` | **Deliverable B (Section 5: Reproducibility)** | Original authors' experimental notebook, debugged and fully executed (130 code cells, 0 errors, 59 embedded figures) to reproduce the paper's benchmark across 8 model families. |
| `nochlezhka_modeling_extensions.ipynb` | **Deliverable C (Section 6b: Extended Modeling)** | Clean, self-contained notebook implementing **5 methodological and policy-relevant extensions** with side-by-side comparisons against the paper's baseline results. |
| `final_with_onehot_types_ALL_221120.csv` | Primary Modeling Dataset | 8,532 social contract records across 3,215 unique clients with 96 extracted features (demographics, case-note indicators, income sources, and 43 one-hot encoded contract types) plus contract status and binary success outcomes. |
| `all_events_anonymized.csv` | Raw Event Log | 127,990 timestamped humanitarian project visits and social contract events per anonymized client ID. |
| `anonymized_contracts.csv` | Contract Metadata | Detailed social contract records including start dates, end dates, and numerical status IDs. |
| `anonymized_ids_map.csv` | Entity Resolution Map | Mapping table linking each client's social contract ID and humanitarian project ID. |
| `contract_status.csv` & `contract_types.csv` | Codebooks | Lookup tables for numerical contract status IDs (`1`–`8`) and contract intervention types (`type_1`–`type_43`). |
| `xgb_rm_CORRECTED.model` & `xgb_ros_CORRECTED.model` | Pre-trained XGBoost Artifacts | Serialized `XGBClassifier` models (undersampled and oversampled) re-saved with complete 96-feature booster metadata. |
| `cb.model` & `cb_ros.model` | Pre-trained CatBoost Artifacts | Serialized `CatBoostClassifier` models (undersampled and oversampled) generated to resolve missing file errors in the original repository. |
| `Appendix.pdf` | Supplementary Material | Original appendix detailing feature definitions and feature-wise missingness (`NaN`) statistics. |

---

## Section 5: Reproducibility Audit — Issues Encountered & Engineering Fixes

While attempting to reproduce the original analyses in `nochlezhka_rel_2.ipynb` using a modern Python 3.11 environment (`pandas 3.0.0`, `numpy 2.4.2`, `scikit-learn 1.9.1`, `imbalanced-learn 0.14.2`, `xgboost 1.6.0`, `catboost 1.2.10`, `torch 2.15.0`), the notebook failed to execute out-of-the-box due to **seven distinct reproducibility defects** spanning missing artifacts, undefined variables, non-deterministic data structures, metric parameter inversion bugs, and API deprecations:

### 1. Undefined Global Random Seed (`SEED`)
* **Problem:** Starting at Cell 29 (`train_test_split(..., random_state=SEED)`), the global constant `SEED` is passed into every data split, resampling step (`RandomUnderSampler`, `RandomOverSampler`), and estimator constructor (`LogisticRegression`, `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `XGBClassifier`, `CatBoostClassifier`), but **`SEED` is never defined anywhere in the notebook**, immediately raising `NameError: name 'SEED' is not defined`.
* **Resolution:** By inspecting the serialized `XGBClassifier` object inside the authors' model artifact (`joblib.load('xgb_rm_CORRECTED.model').get_params()['random_state']`), we verified that the authors' original seed was `42` and explicitly defined `SEED = 42` in Cell 5.

### 2. Missing Pre-trained Model Files (`cb.model`, `cb_ros.model`) & Corrupted Booster Metadata
* **Problem:**
  1. Cell 141 attempts to load pre-trained CatBoost models via `cb.load_model('cb.model')` and `cb_ros.load_model('cb_ros.model')`, neither of which was included in the repository, raising `CatBoostError: Model file doesn't exist: cb.model`.
  2. In Cell 113, loading the provided `xgb_rm_CORRECTED.model` and `xgb_ros_CORRECTED.model` yielded `xgb.get_booster().feature_names = None`, triggering `TypeError: object of type 'NoneType' has no len()` in Cell 114 and `KeyError: None` in Cell 115 (`X_val_t[features_under]`).
* **Resolution:** Trained and serialized `cb.model` and `cb_ros.model` in the repository root, re-serialized `xgb_rm_CORRECTED.model` and `xgb_ros_CORRECTED.model` directly from pandas DataFrames so all 96 feature names are preserved in the booster metadata, and added a fallback (`xgb.get_booster().feature_names or list(X_val_t.columns)`) in Cell 113.

### 3. Non-Deterministic Feature Column Ordering via Python `set()`
* **Problem:** In Cell 28, the authors constructed the feature list using unordered set subtraction:
  ```python
  x_columns_t = list(set(subset.columns) - set(['contract_id', 'client_id', 'contract_success', 'contract_status_id']))
  ```
  Because Python randomizes hash seeds (`PYTHONHASHSEED`) across interpreter sessions, `list(set(...))` permutes the 96 column order on every run. The authors even noted in Markdown Cell 111: *"There is a problem with loading saved model: the feature names get shuffled (had to ask at stackoverflow)"*—unaware that their own `set()` operation in Cell 28 caused the shuffling.
* **Resolution:** Replaced the `set()` subtraction in Cell 28 with an order-preserving list comprehension:
  ```python
  x_columns_t = [c for c in subset.columns if c not in ['contract_id', 'client_id', 'contract_success', 'contract_status_id']]
  ```

### 4. Inverted `y_true` and `y_pred` Arguments in `precision_score` & `recall_score` + CatBoost Prediction Bug
* **Problem:**
  1. Across every evaluation cell in `nochlezhka_rel_2.ipynb` (Cells 52, 53, 60, 61, 67, 68, 77, 78, 83, 84, 116, 117, 128, 129, 142, 143, 145, 147), the authors wrote:
     ```python
     print('Precision =', precision_score(pred, y_val_t))
     print('Recall =', recall_score(pred, y_val_t))
     ```
     In `scikit-learn`, the signature is `precision_score(y_true, y_pred)`. Passing `(y_pred, y_true)` swaps False Positives (`FP`) and False Negatives (`FN`), meaning **every printed "Precision" in the notebook is actually Recall, and every printed "Recall" is actually Precision** (which can be verified by comparing the printed lines against `classification_report(y_val_t, pred)` directly below them).
  2. In Cells 143 and 147, when evaluating the oversampled CatBoost model `cb_ros`, the authors mistakenly called `cb_ros_pred = cb.predict(X_val_t)` (using the undersampled model `cb`), leading them to write in Markdown Cell 144: *"Results seem identical. Probably, the same file was saved under different names."*
* **Resolution:** Fixed Cells 143 and 147 to call `cb_ros.predict(X_val_t)`, and in `nochlezhka_modeling_extensions.ipynb` we report both standard `(y_true, y_pred)` metrics and the paper's inverted metrics clearly.

### 5. Silent Cross-Validation Degradation in `scikit-learn >= 1.6` with `XGBClassifier`
* **Problem:** In `scikit-learn >= 1.6`, `is_classifier()` checks `__sklearn_tags__()`. Because `xgboost 1.6.0` predates this API change, `is_classifier(XGBClassifier())` returns `False`. Consequently, `cross_val_score(clf, X, y, cv=3)` in `hyperopt_xgb_score` silently fell back from `StratifiedKFold(3)` to unshuffled `KFold(3)`. Since `RandomUnderSampler` outputs labels sorted by class (`0, ..., 0, 1, ..., 1`), unshuffled 3-fold splitting created single-class folds and collapsed cross-validation scores from ~0.77 to ~0.55.
* **Resolution:** Explicitly passed `cv=StratifiedKFold(n_splits=3), scoring='accuracy'` to `cross_val_score` in Cells 10 and 123.

### 6. Legacy `!pip install` Pinning & PyTorch `Series` Tensor Construction
* **Problem:** Cells 1–3 forced `!pip install matplotlib==3.1.3` (from 2020), which fails to compile on Python 3.11+. In Cell 91, passing resampled pandas `Series` directly to `torch.FloatTensor(y_res_train_std)` fails when Series indices are non-contiguous.
* **Resolution:** Commented out inline `!pip install` magic commands and wrapped inputs in `np.asarray(...)` prior to `torch.FloatTensor(...)`.

### 7. Removed `DataFrame.append` and `value_counts()[0]` Label Indexing in `pandas 2.0+ / 3.0`
* **Problem:** Cell 151 called `no_client_t.append(pd.Series(), ignore_index=True)` (removed in `pandas 2.0`), raising `AttributeError`. Cell 156 indexed `X_t[column].value_counts()[0]`, which raises `KeyError: 0` on `gender` (encoded as `1 = Male`, `2 = Female`, with no `0` key).
* **Resolution:** Replaced `DataFrame.append` with `pd.DataFrame(np.nan, index=[0], columns=x_columns_t)` in Cell 151, and replaced `.value_counts()[0]` with `.value_counts().get(0, 0)` in Cell 156.

### Baseline Reproduced Results (`nochlezhka_rel_2.ipynb`, N = 6,349, Validation N = 1,270)

> **Note on Precision & Recall Columns:** Below we report the true standard `scikit-learn` metrics `(y_true, y_pred)` alongside the paper's inverted printout `(y_pred, y_true)` in parentheses where they differ.

| Model Architecture | Sampling Strategy | Accuracy | True Precision (Pos=1) | True Recall / TPR (Pos=1) | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Undersampling | 0.6346 | 0.7671 *(Paper: 0.6274)* | 0.6274 *(Paper: 0.7671)* | 0.6903 |
| **Logistic Regression** | Oversampling | 0.6370 | 0.7705 *(Paper: 0.6274)* | 0.6274 *(Paper: 0.7705)* | 0.6916 |
| **SVM (Linear Kernel)** | Undersampling | 0.6378 | 0.7692 *(Paper: 0.6311)* | 0.6311 *(Paper: 0.7692)* | 0.6933 |
| **SVM (Linear Kernel)** | Oversampling | 0.6315 | 0.7825 *(Paper: 0.5983)* | 0.5983 *(Paper: 0.7825)* | 0.6781 |
| **Decision Tree** | Undersampling | 0.7228 | 0.8296 *(Paper: 0.7209)* | 0.7209 *(Paper: 0.8296)* | 0.7714 |
| **Decision Tree** | Oversampling | 0.7693 | 0.8435 *(Paper: 0.7913)* | 0.7913 *(Paper: 0.8435)* | 0.8165 |
| **k-Nearest Neighbors** | Undersampling | 0.6346 | 0.7778 *(Paper: 0.6117)* | 0.6117 *(Paper: 0.7778)* | 0.6848 |
| **k-Nearest Neighbors** | Oversampling | 0.6449 | 0.7739 *(Paper: 0.6396)* | 0.6396 *(Paper: 0.7739)* | 0.7003 |
| **Random Forest** | Undersampling | 0.7709 | 0.8791 *(Paper: 0.7500)* | 0.7500 *(Paper: 0.8791)* | 0.8094 |
| **Random Forest** | Oversampling | 0.8142 | 0.8551 *(Paper: 0.8592)* | 0.8592 *(Paper: 0.8551)* | 0.8571 |
| **FCNN (PyTorch)** | Undersampling | 0.6976 | 0.8129 | 0.6869 | 0.7446 |
| **FCNN (PyTorch)** | Oversampling | 0.7307 | 0.7932 | 0.7942 | 0.7937 |
| **XGBoost** | Undersampling | 0.7937 | 0.8914 *(Paper: 0.7767)* | 0.7767 *(Paper: 0.8914)* | 0.8301 |
| **XGBoost** | Oversampling | **0.8339** | **0.8761** *(Paper: 0.8665)* | **0.8665** *(Paper: 0.8761)* | **0.8713** |
| **CatBoost** | Undersampling | 0.7551 | 0.8680 *(Paper: 0.7342)* | 0.7342 *(Paper: 0.8680)* | 0.7955 |
| **CatBoost** | Oversampling | 0.7827 | 0.8568 *(Paper: 0.7985)* | 0.7985 *(Paper: 0.8568)* | 0.8266 |

---

## Section 6: Extended Modeling Analyses (`nochlezhka_modeling_extensions.ipynb`)

### Section 6a: Researcher Decision Process & Unreported Analyses in the Original Code
Inspecting `nochlezhka_rel_2.ipynb` reveals substantial evidence of exploratory decisions and unpublished experiments left behind in comments and unused code blocks:
1. **Selective Cohort Filtering (`subset` vs. Full Dataset):** In Cell 25, the authors filtered the dataset from 8,532 total contracts down to 6,349 (`contract_status_id.isin([2, 3, 5, 7, 8])`), silently discarding 1,317 contracts with status `4` (`Partial - other`) and 458 contracts with status `6` (`Fail - other`). Including those 1,775 grey-zone failures reduces the overall contract success rate from 64.88% to 50.70% and lowers peak F1 scores by nearly 8 percentage points.
2. **Commented-Out Hyperparameter Trials (Cell 125):** Cell 125 contains five commented-out parameter dictionaries with inline confusion-matrix annotations (e.g., `#best_params = {'eta': 0.25, 'max_depth': 14, 'n_estimators': 585}# 81 80 81 371 656 1027`), showing that the authors manually tuned `XGBClassifier` against the validation set (`X_val_t`) rather than selecting hyperparameters strictly from inner cross-validation.
3. **Unused Client-Specific Counterfactual Recommender (`find_diff` & `find_important_features`, Cells 12–14 & 151–168):** The authors wrote functions to construct a synthetic "Typical Client" (`typical_client_t`) and perturb unknown (`NaN`) features between `0` and `1` (`find_diff`), but never executed `find_important_features` in the final notebook because tree models produced unstable counterfactual swings on sparse one-hot contract types.

---

### Section 6b: Five Extended Modeling Analyses & Side-by-Side Results

All five extended analyses are fully implemented, commented, and visualized in [`nochlezhka_modeling_extensions.ipynb`](./nochlezhka_modeling_extensions.ipynb).

#### Master Side-by-Side Comparison Across Experimental Settings (Oversampled Models)

| Experimental Setting | Sample Size (N) | Val Split Design | Feature Set | XGBoost Acc | XGBoost F1 | Random Forest Acc | Random Forest F1 | CatBoost Acc | CatBoost F1 | Logistic Reg Acc | Logistic Reg F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Paper Baseline** | 6,349 | Row Split | All 96 Feats (Leaked Impute) | **0.8339** | **0.8713** | 0.8142 | 0.8571 | 0.7827 | 0.8266 | 0.6370 | 0.6916 |
| **1. Extension 1: Leak-Free Preprocessing** | 6,349 | Row Split | All 96 Feats (Train-Only Fit) | 0.8339 | 0.8713 | 0.8157 | 0.8584 | 0.7827 | 0.8266 | 0.6370 | 0.6916 |
| **2. Extension 2: Client-Level Group Split** | 6,349 | **`GroupShuffleSplit` (`client_id`)** | All 96 Feats (Leak-Free) | **0.6785** *(-15.5%)* | **0.7694** *(-10.2%)* | **0.6971** *(-11.9%)* | **0.7902** *(-6.8%)* | **0.6696** *(-11.3%)* | **0.7520** *(-7.5%)* | 0.6341 *(-0.3%)* | 0.7010 *(+0.9%)* |
| **3. Extension 3: Ex-Ante Features Only** | 6,349 | Row Split | **94 Feats (No Absence/Disturb.)** | 0.8409 *(+0.7%)* | 0.8771 *(+0.6%)* | 0.8150 *(-0.1%)* | 0.8582 *(0.0%)* | 0.7866 *(+0.4%)* | 0.8286 *(+0.2%)* | 0.6299 *(-0.7%)* | 0.6850 *(-0.7%)* |
| **3b. Extension 2 + 3 Combined (Strict Ex-Ante + Unseen Clients)** | 6,349 | **`GroupShuffleSplit` (`client_id`)** | **94 Feats (No Absence/Disturb.)** | **0.6672** *(-16.7%)* | **0.7632** *(-10.8%)* | **0.6834** *(-13.1%)* | **0.7775** *(-8.0%)* | **0.6753** *(-10.7%)* | **0.7590** *(-6.8%)* | 0.6268 *(-1.0%)* | 0.6948 *(+0.3%)* |
| **4. Extension 4: Full Resolved Cohort** | **8,124** | Row Split | All 96 Feats (Statuses 2–8) | **0.7963** *(-3.8%)* | **0.7933** *(-7.8%)* | **0.7631** *(-5.1%)* | **0.7637** *(-9.3%)* | **0.7582** *(-2.5%)* | **0.7545** *(-7.2%)* | 0.6375 *(+0.1%)* | 0.6348 *(-5.7%)* |

---

#### Detailed Breakdown of the Five Extended Analyses

##### Extension 1: Eliminating Preprocessing Data Leakage (Train-Only Imputation & Scaling)
* **Modification:** In `nochlezhka_rel_2.ipynb` (Cells 40–45), the authors computed column means `X_t[rest].fillna(X_t[rest].mean())` and applied `StandardScaler().fit_transform(X_nonans)` across all 6,349 rows *before* calling `train_test_split`. We refactored the pipeline to split `X_train` and `X_val` first, estimating imputation means and scaler parameters strictly on `X_train`.
* **Findings:** Fixing preprocessing leakage alters F1 scores by ±0.0009 to ±0.0021 across `LogisticRegression`, `SVC`, `KNeighborsClassifier`, and `RandomForestClassifier`. While the numerical shift is modest because global means across 6,349 rows closely approximate training means across 5,079 rows, isolating validation statistics is essential for methodological integrity.

##### Extension 2: Entity-Level Group Splitting by `client_id` (`GroupShuffleSplit`)
* **Modification:** The 6,349 contracts in `subset` belong to only 2,754 unique clients (2.31 contracts per client). Auditing the authors' row-level `train_test_split` revealed that **68.76% (656 / 954) of validation clients also appeared in the training set**, accounting for **73.54% (934 / 1,270) of all validation contracts**. We replaced `train_test_split` with `GroupShuffleSplit(test_size=0.2, random_state=42)` grouped by `client_id` (0% client overlap).
* **Findings:**
  * Under `GroupShuffleSplit`, **XGBoost (Oversampled) accuracy plummets by -15.53 percentage points (0.8339 → 0.6785) and F1 drops by -0.1019 (0.8713 → 0.7694)**. Random Forest and CatBoost suffer similar drops (-11.87% and -11.30% in accuracy), whereas linear Logistic Regression remains completely stable (0.6370 → 0.6341).
  * Disaggregating the original validation set confirms a **memorization gap of +20.71% accuracy**: on the 934 validation contracts belonging to clients seen during training, XGBoost achieves **88.87% accuracy (F1 = 0.9105)**, whereas on the 336 contracts belonging to truly unseen clients, accuracy is only **68.15% (F1 = 0.7757)**. Thus, the paper's headline 83.4% accuracy largely reflects **memorization of repeat clients' static demographic fingerprints** rather than generalizable contract dynamics.

##### Extension 3: Removing Post-Hoc Target Leakage Features (`comments_absence` & `comments_disturbance`)
* **Modification:** Client absence (`status 7` and `status 8`) accounts for 81.4% of all failed contracts in `subset`. Auditing `comments_absence` shows that it occurs at a **5.57%–8.14% rate in absence-terminated contracts** vs. **1.73% in completed contracts**. Because case notes documenting client absence or shelter rule disturbances (`comments_disturbance`) are recorded *during* contract execution rather than at intake, we removed both features (96 → 94 features).
* **Findings:** Removing post-hoc process features reduces Logistic Regression F1 by -0.0066 (0.6916 → 0.6850) and reduces group-split XGBoost F1 from 0.7694 to 0.7632 (and group-split Random Forest F1 from 0.7902 to 0.7775), providing an honest measure of *ex-ante* intake predictability.

##### Extension 4: Expanding Sample Inclusion to All Resolved Contracts (Adding Status `4` & `6`)
* **Modification:** We re-included the 1,775 resolved contracts omitted by the authors (`4: Partial - other`, N = 1,317; `6: Fail - other`, N = 458), expanding the dataset from N = 6,349 (64.88% success) to N = 8,124 (50.70% success, excluding only `1: In progress`).
* **Findings:** When forced to classify complex, non-voluntary failures (`other` reasons), **F1 scores drop sharply across every model family**: XGBoost (OS) F1 falls by **-0.0780 (0.8713 → 0.7933)**, Random Forest (OS) F1 falls by **-0.0947 (0.8584 → 0.7637)**, CatBoost (OS) F1 falls by **-0.0721 (0.8266 → 0.7545)**, and Logistic Regression F1 falls by **-0.0568 (0.6916 → 0.6348)**.

##### Extension 5: Subgroup Disaggregated Evaluation & Algorithmic Fairness Audit
* **Modification:** We audited the validation predictions of XGBoost (Oversampled) across **Gender**, **Age Brackets**, **Housing Status**, and **Chemical Addiction Disclosure**, tracking **False Positive Rate (FPR — Missed Failure Risk)** and **False Negative Rate (FNR — Unjust Pessimism / Rationing Risk)**:

| Subgroup Slice | Val N | Base Success Rate | Accuracy | Precision | Recall (TPR) | F1 Score | FPR (Missed Risk) | FNR (Unjust Pessimism) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Validation Set** | 1,270 | 0.6488 | 0.8339 | 0.8761 | 0.8665 | 0.8713 | 0.2265 | 0.1335 | 0.8957 |
| **Gender: Male (`gender=1`)** | 1,000 | 0.6370 | 0.8350 | 0.8746 | 0.8650 | 0.8698 | 0.2176 | 0.1350 | 0.8999 |
| **Gender: Female (`gender=2`)** | 270 | 0.6926 | 0.8296 | 0.8811 | 0.8717 | 0.8763 | **0.2651** | 0.1283 | 0.8779 |
| **Age: Youth (`<35 yrs`)** | 139 | 0.6547 | 0.8633 | 0.9286 | 0.8571 | 0.8914 | 0.1250 | **0.1429** | 0.9398 |
| **Age: Middle-Aged (`35–54 yrs`)** | 667 | 0.6552 | 0.8291 | 0.8696 | 0.8696 | 0.8696 | **0.2478** | 0.1304 | 0.8862 |
| **Age: Older Adults (`55+ yrs`)** | 464 | 0.6379 | 0.8319 | 0.8707 | 0.8649 | 0.8678 | 0.2262 | 0.1351 | 0.8948 |
| **Housing: Currently Homeless** | 1,263 | 0.6477 | 0.8329 | 0.8752 | 0.8655 | 0.8703 | 0.2270 | 0.1345 | 0.8951 |
| **Chem Addiction: Confirmed / Refused** | 81 | 0.7531 | 0.8519 | 0.9016 | 0.9016 | 0.9016 | **0.3000** | 0.0984 | 0.9008 |
| **Chem Addiction: Denies Addiction** | 122 | 0.7049 | 0.8279 | 0.8736 | 0.8837 | 0.8786 | **0.3056** | 0.1163 | 0.8345 |

* **Fairness & Policy Takeaways:**
  1. **Elevated Missed-Risk Rate (FPR) for Women (26.51% vs. 21.76% for Men):** Because female clients have a higher baseline completion rate (69.26% vs. 63.70%), the model over-predicts success for women, failing to flag 26.51% of women who ultimately drop out (`ROC-AUC` is also lower for women at 0.8779 vs. 0.8999 for men).
  2. **High False Positive Rate Among Clients Answering Chemical Addiction Questions (30.00%–30.56%):** Clients with recorded chemical addiction assessments experience a ~30% false positive rate, meaning nearly one-third of unsuccessful contracts in this vulnerable group are misclassified as low-risk.
  3. **Higher False Negative Rate (FNR) for Youth (14.29%):** Young homeless clients (`<35` years) face the highest rate of unjust pessimism (FNR = 14.29%), where capable youth who actually succeed are misclassified as failures. In practice, predictive scores must **never** be used to ration or deny shelter/rehabilitation contracts, but rather to trigger proactive caseworker check-ins.

---

## Key Takeaways: How the Extended Analyses Fundamentally Change the Paper's Conclusions

The central finding of our critical re-examination is that **the original paper presents an artificially inflated and methodologically fragile baseline (headline 83.39% accuracy, 0.8713 F1)**, while our extended analyses reveal the **true, sober reality of algorithmic prediction in real-world homeless rehabilitation (cold-start accuracy of 66.72%–69.71%, 0.7632–0.7902 F1)**.

Below is a structured synthesis of the **six fundamental differences** between the paper's reproduced baseline and our extended empirical findings:

### 1. The Core Comparison Matrix: Baseline Illusion vs. Extended Empirical Reality

| Methodological Dimension | Original Paper Baseline (Reproduced) | Extended Analyses (Deliverable C) | Critical Difference & Social Impact Takeaway |
| :--- | :--- | :--- | :--- |
| **A. Repeat-Client Generalization** | **83.39% Accuracy / 0.8713 F1** (XGBoost OS)<br>Evaluated on row-split validation set | **67.85% Accuracy / 0.7694 F1** (XGBoost OS)<br>**69.71% Accuracy / 0.7902 F1** (Random Forest OS)<br>Evaluated via `GroupShuffleSplit` on `client_id` | **A massive 15.5% accuracy collapse.** In the paper's split, **73.54% of validation contracts belong to repeat clients already seen in training**. Tree models achieve 88.87% on seen clients vs. only 68.15% on brand-new clients—proving the baseline is memorizing static individual client traits rather than generalizable contract mechanisms. Random Forest actually overtakes XGBoost on unseen clients. |
| **B. Target Cohort Inclusion** | **64.88% Base Success Rate**<br>Artificially curated subset (N = 6,349)<br>Omits status `4` and `6` | **50.70% Base Success Rate**<br>Full resolved cohort (N = 8,124)<br>XGBoost F1 drops from 0.8713 to **0.7933**<br>Random Forest F1 drops from 0.8584 to **0.7637** | The authors discarded 1,775 resolved contracts (44.3% of all non-successful outcomes) attributed to external or administrative obstacles. Restoring these messy, real-world failures reveals that actual contract success is essentially a coin-toss (50.7%), and model discriminative power drops by 8–9 F1 points. |
| **C. Temporal & Causal Feasibility** | Uses **96 features**, including post-hoc case notes (`comments_absence`) | Uses **94 ex-ante features**, strictly dropping `comments_absence` & `comments_disturbance` | Because client absence defines 81.4% of paper failures, conditioning on social worker notes documenting absence constitutes target leakage. Removing post-hoc notes yields cold-start intake accuracy of **66.72%** (XGBoost) and **68.34%** (Random Forest). |
| **D. Preprocessing Leakage Audit** | **Leaked Pipeline** (`fillna` and `StandardScaler` fit on full `X_t` before split) | **Strict Leak-Free Pipeline** (imputer and scaler fit strictly on `X_train`) | Metric shifts are negligible (±0.001 to 0.002 F1), successfully **ruling out data transformation leakage** as the cause of high performance and isolating repeat-client memorization and cohort filtering as the true drivers. |
| **E. Algorithmic Fairness & Subgroup Equity** | Reports only a **single monolithic aggregate accuracy (83.39%)**; assumes uniform equity across groups | Disaggregates performance across **Gender, Age, Housing, and Chemical Addiction**: | The aggregate score masks dangerous error disparities across vulnerable groups: |
| **F. Gender Equity** | Unreported | **FPR = 26.51% for Women vs. 21.76% for Men** (ROC-AUC 0.8779 vs. 0.8999) | The model is over-optimistic for female clients (due to higher baseline success, 69.3% vs. 63.7%), failing to flag over one-fourth of women who drop out. |
| **G. Chemical Addiction Disclosure** | Unreported | **FPR = 30.00%–30.56%** for clients with addiction interview records | Nearly one-third of non-completed contracts among this highly vulnerable group are misclassified as low-risk, risking systemic under-support. |
| **H. Age Disparity** | Unreported | **FNR = 14.29% for Youth (`<35` years)** vs. 13.04% for middle-aged | Young clients suffer the highest rate of unjust pessimism (capable youth predicted to fail), showing why predictive models must never gatekeep access. |
| **I. Metric Parameter Inversion Bug** | Printed Precision: `0.8665`<br>Printed Recall: `0.8761` | Standard Precision: **`0.8761`**<br>Standard Recall: **`0.8665`** | Author passed `(y_pred, y_true)` instead of `(y_true, y_pred)` across every cell, inverting Precision and Recall throughout the original paper code. |

---

### 2. In-Depth Narrative Breakdown: What This Means for Social Impact & Practice

#### Insight 1: Deconstructing the "83.4% Accuracy" Myth — Memorization vs. Learning
In social service settings, predictive tools are intended to assist caseworkers when a **new, unfamiliar client** enters a shelter. The original paper claimed an impressive 83.4% accuracy, giving the false impression that machine learning can reliably forecast rehabilitation outcomes at intake.

However, Extension 2 reveals that this performance is largely an artifact of **repeat-client leakage**:
* A client who signs 4 consecutive contracts has their demographic profile (birthplace, age, disability status, penal history, family background) repeated across 4 rows.
* When rows are split randomly, the model encounters this exact same individual in both training and testing.
* On the 73.54% of validation contracts belonging to "seen" individuals, XGBoost achieves **88.87% accuracy (F1 = 0.9105)**. But on the 26.46% of contracts signed by genuinely new, unseen individuals, performance collapses to **68.15% accuracy (F1 = 0.7757)**—a **20.71% memorization gap**.
* When trained under a strict `GroupShuffleSplit`, **Random Forest (69.71% Acc, 0.7902 F1) outperforms XGBoost (67.85% Acc, 0.7694 F1)**, demonstrating that simpler ensembles with aggressive bootstrap bagging generalize better to new individuals than heavily boosted gradient trees.

#### Insight 2: The Fallacy of the Curated Sample — Homelessness Rehabilitation is Messy
The original paper created a curated cohort where nearly two-thirds (64.88%) of contracts succeeded. They achieved this by quietly removing 1,775 contracts with status `4` (`Partial - other`) and `6` (`Fail - other`).

In reality, homelessness rehabilitation takes place in an unpredictable social environment: clients are hospitalized, shelters run out of beds, bureaucratic agencies delay passport issuance, or external employers default on wages. When these 1,775 non-voluntary failures are re-integrated into the analysis (Extension 4):
* The true contract completion rate drops to **50.70%** (virtually a coin flip).
* Model performance drops across every single architecture, with XGBoost F1 falling from **0.8713 to 0.7933** and Random Forest F1 dropping from **0.8584 to 0.7637**.
* This proves that the paper's high accuracy was dependent on filtering out the most challenging, realistic edge cases faced by social workers.

#### Insight 3: The Ethical Imperative — Models for Support Allocation, Never for Gatekeeping
Extension 5 delivers the most critical takeaway for policy and practice:
1. **Unjust Pessimism Against Youth (14.29% FNR):** If an algorithm were used to decide which clients receive scarce transitional housing slots, young homeless clients (`<35` years old) would be disproportionately harmed—14.29% of young people who would have successfully completed their contract would be denied services due to algorithmic pessimism.
2. **False Security for Women & Chemical Addiction Clients (~26%–30% FPR):** Conversely, women and individuals dealing with substance dependency experience high false positive rates. A caseworker relying on algorithmic risk scores might assume these clients are "safe" and withdraw intensive check-ins, allowing them to quietly slip through the cracks.
3. **Policy Conclusion:** Risk scores should **never be used as a gatekeeping mechanism or hurdle to service access**. Instead, they should function exclusively as an **internal caseworker decision-support radar**—directing additional case conferences, legal advocacy, and material aid to contracts flagged as fragile.
