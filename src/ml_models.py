import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

def train_baseline_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains baseline ML models (RandomForest and GradientBoosting).
    Calculates metrics: Precision, Recall, F1-Score, ROC-AUC.
    Performs error analysis on false positives and false negatives.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = {}

    # 1. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)

    y_val_pred_rf = rf.predict(X_val)
    y_val_proba_rf = rf.predict_proba(X_val)[:, 1]

    y_test_pred_rf = rf.predict(X_test)
    y_test_proba_rf = rf.predict_proba(X_test)[:, 1]

    rf_metrics = {
        'val_precision': round(precision_score(y_val, y_val_pred_rf, zero_division=0), 4),
        'val_recall': round(recall_score(y_val, y_val_pred_rf, zero_division=0), 4),
        'val_f1': round(f1_score(y_val, y_val_pred_rf, zero_division=0), 4),
        'val_roc_auc': round(roc_auc_score(y_val, y_val_proba_rf), 4),
        'test_precision': round(precision_score(y_test, y_test_pred_rf, zero_division=0), 4),
        'test_recall': round(recall_score(y_test, y_test_pred_rf, zero_division=0), 4),
        'test_f1': round(f1_score(y_test, y_test_pred_rf, zero_division=0), 4),
        'test_roc_auc': round(roc_auc_score(y_test, y_test_proba_rf), 4)
    }

    # 2. Gradient Boosting Classifier
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)

    y_test_pred_gb = gb.predict(X_test)
    y_test_proba_gb = gb.predict_proba(X_test)[:, 1]

    gb_metrics = {
        'test_precision': round(precision_score(y_test, y_test_pred_gb, zero_division=0), 4),
        'test_recall': round(recall_score(y_test, y_test_pred_gb, zero_division=0), 4),
        'test_f1': round(f1_score(y_test, y_test_pred_gb, zero_division=0), 4),
        'test_roc_auc': round(roc_auc_score(y_test, y_test_proba_gb), 4)
    }

    # Error Analysis
    cm_rf = confusion_matrix(y_test, y_test_pred_rf)
    tn, fp, fn, tp = cm_rf.ravel() if cm_rf.size == 4 else (0,0,0,0)
    error_analysis = {
        'confusion_matrix': cm_rf.tolist(),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'explanation': f"Error Analysis: {fn} missed failures (False Negatives - critical risk) and {fp} false alarms (False Positives)."
    }

    # Save best baseline model
    model_path = os.path.join(MODEL_DIR, 'baseline_rf.joblib')
    joblib.dump(rf, model_path)

    results['RandomForest'] = {'model': rf, 'metrics': rf_metrics}
    results['GradientBoosting'] = {'model': gb, 'metrics': gb_metrics}
    results['error_analysis'] = error_analysis
    results['feature_importance'] = dict(zip(X_train.columns, rf.feature_importances_))

    return results

if __name__ == '__main__':
    from data_engineering import generate_synthetic_factory_data, clean_and_preprocess_telemetry, get_train_val_test_split
    generate_synthetic_factory_data()
    raw = pd.read_csv('../data/raw/machine_sensors.csv')
    cleaned = clean_and_preprocess_telemetry(raw)
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), cols = get_train_val_test_split(cleaned)
    res = train_baseline_models(X_tr, y_tr, X_v, y_v, X_te, y_te)
    print("Baseline models trained successfully!")
    print("Random Forest Metrics:", res['RandomForest']['metrics'])
