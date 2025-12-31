# 🛡️ High-Fidelity Behavioral Fraud Detection: IEEE-CIS Engine

## 📝 Executive Summary
This project implements an end-to-end machine learning pipeline to detect fraudulent credit card transactions. Moving beyond static data analysis, the system reconstructs user behavior through **Black Box Feature Engineering** and implements a **Percentile-Based Thresholding Strategy** to align model outputs with real-world operational constraints. The final system evolved from a linear Logistic Regression baseline to a high-performance LightGBM ensemble, serialized and ready for deployment.

---

## 🚀 Phase 1: Data Ingestion & Structural Alignment
The project began with a high-dimensional dataset (500k+ rows, 400+ features) split into Transaction and Identity files.

* **Header Synchronization**: Identified and corrected a naming discrepancy where `test_identity` used hyphens (`-`) and `train_identity` used underscores (`_`).
* **Inner Join Merging**: Performed an `inner` join on `TransactionID` to ensure that every record in the final set possessed both financial and technical (identity) attributes.
* **Memory Optimization**: Implemented a custom downcasting function to convert numerical types (e.g., `int64` to `int16`) and strings to `category` dtypes, reducing RAM usage by over 50% without losing information.

---

## 🛠️ Phase 2: Strategic Missing Value Analysis (MNAR)
Instead of arbitrary imputation, the project treated missing data as a predictive signal.

* **MNAR Analysis**: Discovered a **42% relative difference** in fraud rates between records with and without `DeviceInfo`, proving data was **Missing Not At Random**.
* **Feature Pruning**: Dropped 30 columns where missingness showed near-zero statistical variance or predictive value.
* **Binary Indicators**: For the remaining features, a "Safety Net" strategy was used: filling numerical NaNs with training-set medians while simultaneously creating **Missing Indicators** (`_is_missing`) to allow the model to weigh the significance of the data gap.

---

## 🧪 Phase 3: "Black Box" Behavioral Engineering
To provide the model with "historical memory," we engineered features that track transaction velocity and deviation.

* **Entity Reconstruction**: Created a proxy UID using `card1` and `addr1` to simulate unique user tracking.
* **Velocity Ratios**: Engineered ratios like `TransactionAmt_to_card1_mean` and `C-series` ratios to detect sudden spikes in activity relative to a card's typical behavior.
* **Geographic Variety**: Calculated the `nunique` count of `addr1` per `card1` to flag cards being used across anomalous physical locations.
* **Leakage Prevention**: To ensure valid evaluation, all means were calculated on `x_train` and mapped to other sets via dictionary lookups, preventing "future" data from influencing training.



---

## 📉 Phase 4: Modeling & Baseline Analysis
The modeling phase followed a rigorous "Baseline-to-Champion" evolution.

* **Redundancy Filter**: Analyzed the 339 PCA-based V-features and dropped columns with a **correlation > 0.99** to reduce multicollinearity.
* **The Baseline**: Trained a **Logistic Regression** model using a `SAGA` solver and `class_weight='balanced'` to establish a performance floor for the 3.5% minority fraud class.
* **Feature Audit**: Extracted model coefficients to verify that engineered ratios and missingness flags were among the top 20 predictors, validating the engineering phase.

---

## 🏆 Phase 5: The Champion Model (LightGBM)
Transitioned to **LightGBM** to capture high-order, non-linear feature interactions.

* **Performance Leap**: LightGBM utilized leaf-wise growth to identify complex fraud signatures (e.g., high `TransactionAmt` + specific `DeviceType` + anomalous `C-ratio` spikes).
* **Efficiency**: The model achieved higher ROC-AUC in significantly less training time (~2 minutes vs 30+ minutes for Logistic Regression).

---

## 🎯 Phase 6: Threshold Selection & Decision Strategy
Recognizing that ROC-AUC is only a ranking metric, we implemented a real-world decision layer.

* **Percentile-Based Thresholding**: Rather than using a default 0.5 probability, we set a decision threshold at the **98th percentile** to reflect operational constraints.
* **Operational Control**: Only the top 2% of highest-risk transactions were flagged, ensuring a manageable volume for manual investigation teams.
* **Precision-Recall Trade-off**: Validated that LightGBM provided significantly better **Recall** at the 98th percentile compared to the baseline, catching more fraud while maintaining an acceptable false-positive rate.



---

## 📦 Phase 7: Deployment Packaging
The final step involved "freezing" the system for production.

* **Serialization**: Used `joblib` to save the trained LightGBM model (`lightgbm_fraud_model_v1.pkl`).
* **Metadata Consistency**: Packaged feature ordering and evaluation metadata with the model to ensure inference data matches the training distribution exactly.

---

## 🌐 Phase 8: Inference Microservice & API Deployment
To move from research to production, the project implements a real-time REST API capable of scoring individual transactions in milliseconds.

* **Flask Microservice**: Developed a robust `app.py` that serves the LightGBM estimator as a high-concurrency web service.
* **Production Preprocessing (`features.py`)**: Designed a standalone inference engine that mirrors the notebook's complex **"Black Box"** engineering. It handles:
    * **$O(1)$ Behavioral Lookups**: Uses serialized dictionaries to calculate real-time ratios (e.g., `TransactionAmt_to_card1_mean`) without requiring the full training dataset.
    * **Schema Expansion**: Employs `df.reindex` and post-reindex imputation to ensure payloads with missing features are safely expanded to the full 200+ feature set.
* **Strict Metadata Parity**: Resolved the `categorical_feature mismatch` by enforcing identical category mappings and `float32` precision between training and serving environments.
* **Operational Thresholding**: The API applies the optimized **0.19 threshold** as a **"First-Class Object,"** allowing for real-time risk stratification (**Low, Medium, High**) without model retraining.

### 📊 Risk Stratification & Output
The API does not merely return a raw score; it provides a multi-layered risk assessment in the JSON response:
* **Probability**: The raw suspiciousness score generated by the LightGBM estimator.
* **Decision**: A binary `Fraud` or `Not Fraud` label based on the optimized **$0.19$ threshold**.
* **Risk Buckets**: Automated categorization of transactions into three tiers to guide manual investigation priority:
    * 🟢 **Low Risk**: Probability $< 0.10$.
    * 🟡 **Medium Risk**: $0.11 \le \text{Probability} \le 0.50$.
    * 🔴 **High Risk**: Probability $> 0.50$.
---


## 🛠️ Tech Stack
* **Languages**: Python (Pandas, NumPy)
* **ML Frameworks**: Scikit-Learn, LightGBM
* **Serialization**: Joblib
* **Web Framework**: Flask 
* **API Testing**: Postman
* **Data Format**: Parquet
* **Optimization**: Custom downcasting, PCA-based correlation pruning

## 👤 Author
**Ezechukwu Princewill**

* **GitHub**: [https://github.com/princewillezechukwu3-lang]
* **Project**: Fraud Detection Pipeline