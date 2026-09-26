import pandas as pd
import numpy as np
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric
import os

def introduce_drift(df):
    df_drifted = df.copy()
    
    # 1. Add noise to MonthlyCharges
    df_drifted['MonthlyCharges'] = df_drifted['MonthlyCharges'] + np.random.normal(10, 5, size=len(df_drifted))
    
    # 2. Skew Contract (oversample Month-to-month)
    # We'll just randomly change 30% of Two year contracts to Month-to-month
    mask = (df_drifted['Contract'] == 'Two year') & (np.random.rand(len(df_drifted)) < 0.3)
    df_drifted.loc[mask, 'Contract'] = 'Month-to-month'
    
    # 3. Flip Target (Churn)
    mask_churn = np.random.rand(len(df_drifted)) < 0.1
    df_drifted.loc[mask_churn, 'Churn'] = df_drifted.loc[mask_churn, 'Churn'].map({'Yes': 'No', 'No': 'Yes'})
    
    return df_drifted

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco_Churn_Monitoring")
    
    data_path = "Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    # Split 70% reference, 30% current
    reference_data = df.sample(frac=0.7, random_state=42)
    current_data = df.drop(reference_data.index)
    
    # Inject synthetic drift
    current_data_drifted = introduce_drift(current_data)
    
    # Run Evidently Report
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        ColumnDriftMetric(column_name="MonthlyCharges")
    ])
    
    report.run(reference_data=reference_data, current_data=current_data_drifted, column_mapping=None)
    
    # Save HTML
    report_path = "drift_report.html"
    report.save_html(report_path)
    
    with mlflow.start_run(run_name="Evidently_Drift_Report"):
        mlflow.log_artifact(report_path)
        print("Drift report generated and logged to MLflow.")

if __name__ == "__main__":
    main()
