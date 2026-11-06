# "homeless" repo — Reproduction & Experiment Notes

This repository presents the anonymized dataset and experiment results described in the paper *"Rehabilitating Homeless: Dataset and Key Insights"*.

The dataset is based on more than 9 years' experience and data collection of **"Nochlezhka"**, a non-governmental organization helping homeless people.

If you intend to use this dataset for whatever purposes, visit "Nochlezhka" at https://homeless.ru/en/how_to_help/ and donate for the noble cause of helping the homeless.

## Repository Contents

* `Appendix.pdf` — Appendix to the original *"Rehabilitating Homeless: Dataset and Key Insights"* paper containing supplementary tables with detailed descriptions of dataset features and feature-wise NaN statistics.
* `all_events_anonymized.csv` — Events (visits to humanitarian projects or entering social contracts) for each person (anonymized and represented by unique IDs) with dates.
* `anonymized_contracts.csv` — Social contracts for each person (anonymized and represented by unique IDs) in more detail, including contract statuses, start dates, and end dates.
* `anonymized_ids_map.csv` — Representation of each person (each row is a person) by a pair of IDs (one for social contracts, one for humanitarian projects — one person may have either one or both IDs).
* `contract_status.csv` — Description of numerical contract status IDs used in the dataset.
* `contract_types.csv` — Description of numerical contract type IDs used in the dataset.
* `final_with_onehot_types_ALL_221120.csv` — The dataset "body": social contract events with all features extracted from the "Nochlezhka" database, along with contract statuses and outcomes.
* `nochlezhka_rel_2.ipynb` — Experiments described in the paper, updated and fully executed for reproducibility in modern Python environments.
* `xgb_rm_CORRECTED.model` & `xgb_ros_CORRECTED.model` — Pre-trained XGBoost models (with random undersampling and random oversampling, respectively) serialized with full feature metadata.
* `cb.model` & `cb_ros.model` — Pre-trained CatBoost models (with random undersampling and random oversampling, respectively).

---

## Code Reproduction Log: Issues Encountered & Fixes Applied

When reproducing `nochlezhka_rel_2.ipynb` in a modern Python 3.11 data science stack (`pandas 3.0.0`, `numpy 2.4.2`, `scikit-learn 1.9.1`, `imbalanced-learn 0.14.2`, `xgboost 1.6.0`, `catboost 1.2.10`, `torch 2.15.0`, `matplotlib 3.11.2`, `seaborn 0.13.2`), several compatibility issues, missing variables, and bugs in the original notebook caused execution failures. Below is a summary of each issue encountered and how it was resolved.

### 1. Hardcoded Legacy Package Installations (Cells 1–3)
* **Issue:** The opening cells executed `!pip install matplotlib==3.1.3`, `!pip install xgboost=='1.6.0'`, and `!pip install catboost`. Attempting to build `matplotlib==3.1.3` (released in 2020) fails on Python 3.11+ due to C-extension and build-system incompatibilities, and forces unnecessary package downgrades.
* **Fix:** Commented out the inline `!pip install` magic commands so the notebook runs directly against the pre-configured Conda environment without breaking dependencies.

### 2. Missing `SEED` Constant Definition (Cell 5)
* **Issue:** Starting in Cell 29 (`train_test_split(..., random_state=SEED)`), the variable `SEED` is referenced across data splitting, resampling (`RandomUnderSampler`, `RandomOverSampler`), and model initialization (`LogisticRegression`, `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `XGBClassifier`, `CatBoostClassifier`), but `SEED` was never defined in any cell, raising `NameError: name 'SEED' is not defined`.
* **Fix:** Inspected the serialized `XGBClassifier` object in `xgb_rm_CORRECTED.model` (`xgb.get_params()['random_state']`), confirmed the original random seed was `42`, and defined `SEED = 42` in the setup/imports cell (Cell 5).

### 3. `scikit-learn` >= 1.6 Compatibility with `XGBClassifier` in `cross_val_score` (Cells 10 & 123)
* **Issue:** In `scikit-learn >= 1.6`, estimator type detection switched to `__sklearn_tags__()`. Because `xgboost 1.6.0` predates this API change, `sklearn.base.is_classifier(XGBClassifier())` evaluates to `False`. Consequently, `cross_val_score(clf, X, y, cv=3)` inside `hyperopt_xgb_score` fell back to unstratified, unshuffled `KFold(3)` instead of `StratifiedKFold(3)`. Since `RandomUnderSampler` outputs labels sorted by class (`0, ..., 0, 1, ..., 1`), unshuffled 3-fold splitting created folds with single-class distributions and degraded cross-validation scores (~0.55–0.61 instead of ~0.77).
* **Fix:** Imported `StratifiedKFold` from `sklearn.model_selection` and explicitly passed `cv=StratifiedKFold(n_splits=3), scoring='accuracy'` to `cross_val_score` in `hyperopt_xgb_score`.

### 4. Non-Deterministic Feature Ordering & Missing Booster Feature Names (Cells 28, 112–120)
* **Issue:**
  1. Cell 28 constructed the feature list via set subtraction (`list(set(subset.columns) - set(...))`), which produces arbitrary column orderings across Python sessions due to hash randomization.
  2. In Cell 113, `xgb.get_booster().feature_names` returned `None` when loading `xgb_rm_CORRECTED.model` and `xgb_ros_CORRECTED.model`, causing `TypeError: object of type 'NoneType' has no len()` in Cell 114 and `KeyError: None` in Cell 115 (`X_val_t[features_under]`).
* **Fix:**
  1. Replaced the set subtraction in Cell 28 with a deterministic list comprehension preserving the original DataFrame column order:
     ```python
     x_columns_t = [c for c in subset.columns if c not in ['contract_id', 'client_id', 'contract_success', 'contract_status_id']]
     ```
  2. Re-trained and re-serialized `xgb_rm_CORRECTED.model` and `xgb_ros_CORRECTED.model` directly from the pandas DataFrames so `get_booster().feature_names` retains all 96 feature names, and added a safe fallback (`xgb.get_booster().feature_names or list(X_val_t.columns)`) in Cell 113.

### 5. PyTorch `FloatTensor` Conversion from Pandas `Series` (Cell 91)
* **Issue:** Passing resampled pandas `Series` (`y_res_train_std`, `y_ros_train_std`) directly to `torch.FloatTensor(...)` can fail or trigger deprecation warnings in newer versions of `pandas` and `torch` when Series indices are non-contiguous.
* **Fix:** Converted feature matrices and target vectors explicitly via `np.asarray(...)` before passing them to `torch.FloatTensor(...)`.

### 6. Missing Pre-trained CatBoost Model Files & Prediction Variable Typo (Cells 140–147)
* **Issue:**
  1. Cell 141 attempted to load `cb.model` and `cb_ros.model` (`cb.load_model('cb.model')`, `cb_ros.load_model('cb_ros.model')`), which were missing from the repository, raising `CatBoostError: Model file doesn't exist: cb.model`.
  2. In Cells 143 and 147, the oversampled CatBoost model `cb_ros` was evaluated using `cb_ros_pred = cb.predict(X_val_t)` (accidentally calling `.predict()` on the undersampled model `cb` instead of `cb_ros`), which caused the identical metrics noted in Markdown Cell 144.
* **Fix:** Trained and saved both `cb.model` and `cb_ros.model` to the repository root, and fixed Cells 143 and 147 to call `cb_ros_pred = cb_ros.predict(X_val_t)`.

### 7. Deprecated `DataFrame.append` and `Series.value_counts()[0]` Indexing in `pandas` 2.0+ / 3.0 (Cells 151 & 156)
* **Issue:**
  1. Cell 151 used `no_client_t = no_client_t.append(pd.Series(), ignore_index=True)` to initialize a single-row all-NaN DataFrame. `DataFrame.append` was removed in `pandas 2.0`, raising `AttributeError: 'DataFrame' object has no attribute 'append'`.
  2. Cell 156 accessed `X_t[column].value_counts()[0]` and `X_t[column].value_counts()[1]` to find the modal value of binary features. Because the `gender` column is encoded as `1` and `2` (with no `0` key in its `value_counts()` index), label lookup `[0]` raised `KeyError: 0`.
* **Fix:**
  1. Replaced Cell 151 with `no_client_t = pd.DataFrame(np.nan, index=[0], columns=x_columns_t)`.
  2. Replaced direct bracket indexing `[0]` and `[1]` on `value_counts()` in Cell 156 with safe dictionary-style lookups (`vc.get(0, 0)` and `vc.get(1, 0)`), accurately reproducing `ones = ['gender', 'is_homeless']` and the 29 zero-mode columns.

---

## Reproduced Validation Results (`SEED = 42`)

All models were evaluated on the held-out validation split (`1,270` contracts: `446` unsuccessful, `824` successful):

| Model | Sampling Strategy | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Undersampling | 0.7283 | 0.7282 | 0.8244 | 0.7733 |
| **Logistic Regression** | Oversampling | 0.7299 | 0.7330 | 0.8235 | 0.7756 |
| **SVM (Linear Kernel)** | Undersampling | 0.7268 | 0.7209 | 0.8285 | 0.7709 |
| **SVM (Linear Kernel)** | Oversampling | 0.7283 | 0.7209 | 0.8308 | 0.7719 |
| **Decision Tree** | Undersampling | 0.6780 | 0.6723 | 0.7994 | 0.7304 |
| **Decision Tree** | Oversampling | 0.7425 | 0.8083 | 0.7938 | 0.8010 |
| **k-Nearest Neighbors** | Undersampling | 0.6646 | 0.6675 | 0.7835 | 0.7208 |
| **k-Nearest Neighbors** | Oversampling | 0.6709 | 0.7136 | 0.7577 | 0.7350 |
| **Random Forest** | Undersampling | 0.7669 | 0.7524 | 0.8708 | 0.8073 |
| **Random Forest** | Oversampling | 0.8181 | 0.8580 | 0.8611 | 0.8596 |
| **FCNN (PyTorch)** | Undersampling | 0.6976 | 0.8129 | 0.6869 | 0.7446 |
| **FCNN (PyTorch)** | Oversampling | 0.7307 | 0.7932 | 0.7942 | 0.7937 |
| **XGBoost** | Undersampling | 0.7937 | 0.7767 | 0.8914 | 0.8301 |
| **XGBoost** | Oversampling | **0.8339** | **0.8665** | **0.8761** | **0.8713** |
| **CatBoost** | Undersampling | 0.7551 | 0.7342 | 0.8680 | 0.7955 |
| **CatBoost** | Oversampling | 0.7827 | 0.7985 | 0.8568 | 0.8266 |
