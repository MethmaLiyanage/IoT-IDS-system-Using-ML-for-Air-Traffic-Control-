import os
import joblib
import numpy as np
from data_loader import load_and_preprocess_data
from train_random_forest import train_random_forest
from train_decision_tree import train_decision_tree
from evaluate import evaluate_ids
from train_lstm import tune_and_train_lstm
from sklearn.metrics import classification_report
from feature_importance import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================
# Load Dataset
# ======================

X, y, feature_names = load_and_preprocess_data()

print("\nClass Distribution:")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

# ======================
# RF / DT — random stratified split
# ======================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Test class distribution: ", dict(zip(*np.unique(y_test,  return_counts=True))))

# ======================
# LSTM — same random stratified split as RF/DT (fair comparison)
# ======================

print(f"\nLSTM train: {len(y_train):,} | LSTM test: {len(y_test):,}")

LSTM_MODEL_PATH  = os.path.join(RESULTS_DIR, "lstm_model.keras")
LSTM_SCALER_PATH = os.path.join(RESULTS_DIR, "lstm_scaler.pkl")
TUNER_DIR        = os.path.join(os.environ.get('TEMP', 'C:\\Temp'), 'lstm_iot_tuner')

lstm_model, lstm_scaler, X_test_seq, y_test_seq, best_hp = tune_and_train_lstm(
    X_train, y_train,
    X_test,  y_test,
    tuner_dir=TUNER_DIR,
    max_trials=15,
    results_dir=RESULTS_DIR,
)

lstm_model.save(LSTM_MODEL_PATH)
joblib.dump(lstm_scaler, LSTM_SCALER_PATH)
print("LSTM model saved.")

y_pred_lstm = np.argmax(lstm_model.predict(X_test_seq), axis=1)

CLASS_NAMES = ["Normal", "Backdoor", "Password", "DDoS", "Injection", "Ransomware", "XSS", "Scanning"]

print("\n--- LSTM Report ---")
print(classification_report(y_test_seq, y_pred_lstm, target_names=CLASS_NAMES))

lstm_report = classification_report(
    y_test_seq, y_pred_lstm,
    target_names=CLASS_NAMES,
    output_dict=True,
)
pd.DataFrame(lstm_report).transpose().to_csv(os.path.join(RESULTS_DIR, "LSTM_report.csv"))

cm = confusion_matrix(y_test_seq, y_pred_lstm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Norm","Back","Pass","DDoS","Inj","Rans","XSS","Scan"],
            yticklabels=["Norm","Back","Pass","DDoS","Inj","Rans","XSS","Scan"])
plt.title("Confusion Matrix: LSTM")
plt.savefig(os.path.join(RESULTS_DIR, "LSTM_matrix.png"))
plt.close()

# ======================
# Train Random Forest
# ======================

rf_model = train_random_forest(X_train, y_train)
evaluate_ids(rf_model, X_test, y_test, "Random_Forest", RESULTS_DIR)
plot_importance(rf_model, feature_names, results_dir=RESULTS_DIR)
joblib.dump(rf_model, os.path.join(RESULTS_DIR, "rf_model.pkl"))
print("Random Forest model saved.")

# ======================
# Train Decision Tree
# ======================

dt_model = train_decision_tree(X_train, y_train)
evaluate_ids(dt_model, X_test, y_test, "Decision_Tree", RESULTS_DIR)
plot_importance(dt_model, feature_names, results_dir=RESULTS_DIR)
joblib.dump(dt_model, os.path.join(RESULTS_DIR, "dt_model.pkl"))
print("Decision Tree model saved.")
