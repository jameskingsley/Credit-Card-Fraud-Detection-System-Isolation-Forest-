# Credit Card Fraud Detection System (Isolation Forest)

An end-to-end, production-ready fraud detection system that uses unsupervised machine learning (Isolation Forest) to identify rare and abnormal credit card transactions in near real-time.
The project includes model training, threshold tuning, explainability (SHAP), and a Streamlit monitoring dashboard.

#### Project Overview
###### Business Problem

Banks and payment providers must detect fraudulent transactions in near real-time, but face major challenges:

* Fraud labels are rare and highly imbalanced
* Labels are often delayed
* Fraud patterns evolve continuously
* Solution Approach

This project uses Isolation Forest, an unsupervised anomaly detection algorithm, to:
* Learn normal transaction behavior
* Flag rare and abnormal patterns as potential fraud
* Avoid heavy dependence on labeled fraud data
###### Machine Learning Strategy
* Aspect - Choice
* Problem Type - Unsupervised Anomaly Detection
* Model -	Isolation  Forest
* Scaling -	RobustScaler
* Training Data -	Normal (non-fraud) transactions only
* Decision Metric -	Anomaly score
* Thresholding -	Percentile-based tuning
* Explainability -	SHAP (global + local)

 ###### Dataset

* Source: Credit Card Transactions dataset
* Records: 284,807 transactions
* Fraud Rate: ~0.17%

Features:
* Time, Amount
* PCA-transformed features V1 – V28
* Class (0 = normal, 1 = fraud)

###### Data Preparation

* Dropped target label (Class) for unsupervised training
* Applied RobustScaler to handle extreme values
* Trained model only on normal transactions
* Used anomaly scores for inference and evaluation

######  Key Insight:
The model successfully identifies fraud using behavioral deviations, not single-feature rules.

###### Feature Importance

Isolation Forest feature importance was computed by aggregating split frequencies across trees.

###### Top drivers of anomalies:

* Time
* V24, V22, V20
* V16, V14, V10, V17
* V8, V28


###### Explainability (SHAP)
* Global Explainability
* Extreme deviations in PCA features strongly influence fraud detection
* Both unusually high and low values trigger anomalies
* Confirms non-linear fraud behavior
* Local Explainability
* Individual transactions are flagged due to combined effects
* No reliance on simple thresholds (e.g., Amount alone)
This makes the model suitable for regulatory review and auditing.

###### Streamlit Application

The project includes a production-style Streamlit dashboard for inference and monitoring.

Key Features

 * CSV upload for batch predictions
* Threshold tuning via percentile slider
*  Fraud flagging with anomaly scores
* Feature importance visualization
*  SHAP explainability (static, production-safe)
* Fraud rate monitoring


  ###### Why Isolation Forest?

* Works without labeled fraud data
* Handles extreme class imbalance
* Captures non-linear interactions
* Scales well for large transaction volumes
Ideal for early-stage fraud detection and monitoring systems.



###### Author
* James Kingsley Philip
* Data Scientist | Machine Learning | MLOps
