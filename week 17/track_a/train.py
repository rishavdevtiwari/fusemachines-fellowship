import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Basic cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    # Features and target
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    # Preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    return train_test_split(X_processed, y, test_size=0.3, random_state=42), preprocessor

def plot_confusion_matrix(y_true, y_pred, run_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {run_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    filename = f"confusion_matrix_{run_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def plot_roc_curve(y_true, y_proba, run_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {run_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    filename = f"roc_curve_{run_name}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def train_and_log(X_train, X_test, y_train, y_test, model_name, model, params):
    with mlflow.start_run(run_name=model_name) as run:
        # Train model
        model.set_params(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        # Log params and metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # Log artifacts
        cm_file = plot_confusion_matrix(y_test, y_pred, model_name)
        roc_file = plot_roc_curve(y_test, y_proba, model_name)
        
        mlflow.log_artifact(cm_file)
        mlflow.log_artifact(roc_file)
        
        # Log model
        mlflow.sklearn.log_model(model, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        
        # Cleanup artifact files
        os.remove(cm_file)
        os.remove(roc_file)
        
        print(f"Run {model_name} completed.")
        return run.info.run_id, metrics['f1']

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco_Churn_Prediction")
    
    data_path = "Telco-Customer-Churn.csv"
    (X_train, X_test, y_train, y_test), preprocessor = load_and_preprocess_data(data_path)
    
    # Save preprocessor
    joblib.dump(preprocessor, "preprocessor.pkl")
    
    runs = []
    
    # Run 1: Logistic Regression
    params1 = {"C": 1.0, "max_iter": 1000}
    run1, f1_1 = train_and_log(X_train, X_test, y_train, y_test, "LogisticRegression_Default", LogisticRegression(), params1)
    runs.append((run1, f1_1, "LogisticRegression_Default"))
    
    # Run 2: Random Forest
    params2 = {"n_estimators": 50, "max_depth": 5, "random_state": 42}
    run2, f1_2 = train_and_log(X_train, X_test, y_train, y_test, "RandomForest_Shallow", RandomForestClassifier(), params2)
    runs.append((run2, f1_2, "RandomForest_Shallow"))
    
    # Run 3: Random Forest Deeper
    params3 = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    run3, f1_3 = train_and_log(X_train, X_test, y_train, y_test, "RandomForest_Deep", RandomForestClassifier(), params3)
    runs.append((run3, f1_3, "RandomForest_Deep"))
    
    # Register the best model based on F1 Score
    best_run = max(runs, key=lambda x: x[1])
    print(f"Best run: {best_run[2]} with F1: {best_run[1]}")
    
    client = mlflow.tracking.MlflowClient()
    model_name = "TelcoChurnModel"
    
    try:
        client.create_registered_model(model_name)
    except:
        pass # Model might already exist
        
    model_uri = f"runs:/{best_run[0]}/model"
    model_version = client.create_model_version(name=model_name, source=model_uri, run_id=best_run[0])
    
    # Transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )
    print(f"Transitioned version {model_version.version} of {model_name} to Staging")
    
    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    print(f"Transitioned version {model_version.version} of {model_name} to Production")

if __name__ == "__main__":
    main()
