# 🛡️ High-Fidelity Behavioral Fraud Detection: IEEE-CIS Engine

## 📝 Executive Summary
This project implements an end-to-end machine learning pipeline to detect fraudulent credit card transactions. Moving beyond static data analysis, the system reconstructs user behavior through **Black Box Feature Engineering** and evolved from a linear Logistic Regression baseline to an **Optuna-optimized LightGBM Native Booster (v3)**. This journey documents the transition from a "Recall Crisis" (0% detection) to a production-ready API capable of identifying complex, high-velocity fraud signatures through a refined **0.30 Decision Threshold**.

---

## 🚀 Phase 1: Data Ingestion & Structural Alignment
The project began with a high-dimensional dataset (500k+ rows, 400+ features) split into Transaction and Identity files.

* **Header Synchronization**: Identified and corrected a naming discrepancy where `test_identity` used hyphens (`-`) and `train_identity` used underscores (`_`).
* **Inner Join Merging**: Performed an `inner` join on `TransactionID` to ensure every record possessed both financial and technical (identity) attributes.
* **Memory Optimization**: Implemented a custom downcasting function (converting `float64` to `float32` and using `category` dtypes), reducing RAM usage by over 50%.

---

## 🛠️ Phase 2: Strategic Missing Value Analysis (MNAR)
Instead of arbitrary imputation, the project treated missing data as a predictive signal.

* **MNAR Analysis**: Discovered a **42% relative difference** in fraud rates between records with and without `DeviceInfo`, proving data was **Missing Not At Random**.
* **Feature Pruning**: Dropped 30 columns where missingness showed near-zero predictive value.
* **Binary Indicators**: Created **Missing Indicators** (`_is_missing`) to allow the model to weigh the significance of the data gap.

---

## 🧪 Phase 3: "Black Box" Behavioral Engineering
To provide the model with "historical memory," we engineered features that track transaction velocity and deviation.

* **Entity Reconstruction**: Created a proxy UID using `card1` and `addr1` to simulate unique user tracking.
* **Velocity Ratios**: Engineered ratios like `TransactionAmt_to_card1_mean` to detect sudden spikes in activity.
* **Geographic Variety**: Calculated the `nunique` count of `addr1` per `card1` to flag cards used across anomalous physical locations.
* **Leakage Prevention**: All means were calculated on `x_train` only and mapped to test sets to prevent data leakage.

---

## 📉 Phase 4: Modeling & Baseline Analysis
The modeling phase followed a rigorous "Baseline-to-Champion" evolution.

* **Redundancy Filter**: Analyzed 339 PCA-based V-features and dropped columns with a **correlation > 0.99**.
* **The Baseline**: Trained a **Logistic Regression** model using a `SAGA` solver and `class_weight='balanced'` to establish a performance floor.
* **Feature Audit**: Extracted coefficients to verify that engineered ratios were among the top 20 predictors.

---

## 🎯 Phase 5: Threshold Selection & Initial Strategy
Recognizing that ROC-AUC is only a ranking metric, we implemented a real-world decision layer.

* **Percentile-Based Thresholding**: Initially, we set a decision threshold at the **98th percentile** to reflect operational constraints, flagging only the top 2% of highest-risk transactions.
* **Initial 0.19 Calibration**: In early iterations of the API, a manual threshold of **0.19** was tested to widen the net for fraud detection.

---

## 🔬 Phase 6: The "Recall Crisis" & Advanced Engineering (v3)
Upon deeper evaluation, initial models struggled with high-velocity fraud. We re-entered the engineering phase to create "Identity Velocity" indicators:

* **$C1\_per\_Amt$**: A critical ratio flagging "card testing" (high address attempts for low amounts).
* **$is\_new\_user$**: A binary flag ($C13 \le 1$) targeting the high-risk "cold-start" population.
* **Ablation Studies**: Systematically removed these features to prove their worth; the `is_new_user` flag was responsible for a 12% lift in Recall.

---

## 🤖 Phase 7: Bayesian Optimization & Threshold Refinement
To maximize the system's ability to catch fraud, we moved to the **LightGBM Native API** and automated hyperparameter tuning.

* **Optuna Study**: Conducted multiple trials to maximize the **F1-Score**, specifically tuning `scale_pos_weight` and `num_leaves`.
* **Final Threshold (0.30)**: Following extensive Postman validation and error analysis, the decision boundary was moved from **0.19 to 0.30**. This change significantly improved the precision of the "High Risk" bucket while maintaining a high recall for optimized fraud capture.

---

## 🌐 Phase 8: Inference Microservice & API Deployment
The final step involved "freezing" the **Model v3** system for a real-time REST API.

* **Flask Microservice**: Serves the booster as a high-concurrency web service.
* **Production Preprocessing (`features.py`)**: 
    * **O(1) Behavioral Lookups**: Uses serialized dictionaries for real-time ratio calculations.
    * **Schema Expansion**: Enforces identical **555-feature alignment** via `lgb_feature_metadata_v3.joblib`.
    * **1D Scalarization**: Handles the native booster's array output to prevent "Python scalar" conversion errors.

### 📊 Risk Stratification Output
* **Probability**: Raw suspiciousness score.
* **Decision**: Binary `Fraud` or `Not Fraud` label (Optimized at **0.30**).
* **Risk Buckets**: 🟢 Low ($<0.10$), 🟡 Medium ($0.11-0.50$), 🔴 High ($>0.50$).

---

## 🛠️ Tech Stack
* **Languages/Lib**: Python, Pandas, NumPy, Scikit-Learn, LightGBM, Optuna.
* **Deployment**: Flask, Joblib, Postman.

## 👤 Author
**Ezechukwu Princewill**
* **GitHub**: [https://github.com/princewillezechukwu3-lang]
