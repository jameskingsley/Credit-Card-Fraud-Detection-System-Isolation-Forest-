import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

# Page Config
st.set_page_config(
    page_title="Fraud Detection – Isolation Forest",
    page_icon="💳",
    layout="wide"
)

# Load Model & Scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("iso_forest_creditcard.pkl")
    scaler = joblib.load("scaler_creditcard.pkl")
    return model, scaler

iso_forest, scaler = load_artifacts()

# Sidebar
st.sidebar.title("Controls")

threshold_percentile = st.sidebar.slider(
    "Fraud Threshold Percentile",
    min_value=0.1,
    max_value=5.0,
    value=0.5,
    step=0.1
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Transactions CSV",
    type=["csv"]
)

show_shap = st.sidebar.checkbox("Show SHAP Explainability", value=True)

# Header
st.title(" Credit Card Fraud Detection")
st.markdown("""
**Model:** Isolation Forest (Unsupervised)  
**Objective:** Detect rare & abnormal transactions in near real-time  
""")

# Helper Functions
def predict_anomalies(X_scaled, threshold_percentile):
    anomaly_scores = iso_forest.decision_function(X_scaled)
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    fraud_preds = (anomaly_scores <= threshold).astype(int)
    return anomaly_scores, fraud_preds, threshold

def plot_feature_importance(model, feature_names):
    importances = np.zeros(len(feature_names))
    for tree in model.estimators_:
        importances += np.bincount(
            tree.tree_.feature[tree.tree_.feature >= 0],
            minlength=len(feature_names)
        )
    importances /= importances.sum()

    fi = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })
        .sort_values(by="importance", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi["feature"], fi["importance"])
    ax.invert_yaxis()
    ax.set_title("Top 10 Features Driving Anomalies")
    ax.set_xlabel("Relative Importance")
    st.pyplot(fig)

# Main Logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader(" Uploaded Data Preview")
    st.dataframe(df.head())

    if "Class" in df.columns:
        y_true = df["Class"]
        X = df.drop(columns=["Class"])
    else:
        y_true = None
        X = df.copy()

    feature_names = X.columns.tolist()

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    anomaly_scores, fraud_preds, threshold = predict_anomalies(
        X_scaled, threshold_percentile
    )

    df_results = df.copy()
    df_results["anomaly_score"] = anomaly_scores
    df_results["fraud_prediction"] = fraud_preds

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df_results))
    col2.metric("Flagged as Fraud", int(df_results["fraud_prediction"].sum()))
    col3.metric("Fraud Rate (%)", round(df_results["fraud_prediction"].mean() * 100, 3))

    # Results Table
    st.subheader(" Fraud Predictions")
    st.dataframe(
        df_results.sort_values("anomaly_score").head(100)
    )

    # Evaluation (if labels exist)
    if y_true is not None:
        from sklearn.metrics import classification_report, roc_auc_score

        roc_auc = roc_auc_score(y_true, -anomaly_scores)

        st.subheader("Model Evaluation")
        st.metric("ROC-AUC", round(roc_auc, 4))

        report = classification_report(y_true, fraud_preds, digits=4, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    # Feature Importance
    st.subheader(" Global Feature Importance")
    plot_feature_importance(iso_forest, feature_names)

    # SHAP Explainability
    if show_shap:
        st.subheader(" Explainability (SHAP)")

        explainer = shap.TreeExplainer(iso_forest)
        sample_idx = np.random.choice(X_scaled.shape[0], size=500, replace=False)
        X_sample = X_scaled[sample_idx]

        shap_values = explainer.shap_values(X_sample)

        st.markdown("**Global SHAP Summary**")
        fig_summary = plt.figure()
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig_summary)

        fraud_indices = np.where(fraud_preds == 1)[0]
        if len(fraud_indices) > 0:
            idx = fraud_indices[0]
            st.markdown("**Local Explanation (First Fraud Case)**")
            fig_force = plt.figure()
            shap.force_plot(
                explainer.expected_value,
                explainer.shap_values(X_scaled[idx]),
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            st.pyplot(fig_force)

else:
    st.info("Upload a CSV file to start fraud detection.")
