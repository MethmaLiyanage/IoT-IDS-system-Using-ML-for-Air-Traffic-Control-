import os
import joblib
import numpy as np
from data_loader import load_and_preprocess_data
from train_random_forest import train_random_forest
from train_decision_tree import train_decision_tree
from evaluate import evaluate_ids
from feature_importance import plot_importance
from sklearn.model_selection import train_test_split

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

X, y, feature_names = load_and_preprocess_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest...")
rf_model = train_random_forest(X_train, y_train)
evaluate_ids(rf_model, X_test, y_test, "Random_Forest", RESULTS_DIR)
plot_importance(rf_model, feature_names, results_dir=RESULTS_DIR)
joblib.dump(rf_model, os.path.join(RESULTS_DIR, "rf_model.pkl"))
print("Random Forest saved.")

print("Training Decision Tree...")
dt_model = train_decision_tree(X_train, y_train)
evaluate_ids(dt_model, X_test, y_test, "Decision_Tree", RESULTS_DIR)
plot_importance(dt_model, feature_names, results_dir=RESULTS_DIR)
joblib.dump(dt_model, os.path.join(RESULTS_DIR, "dt_model.pkl"))
print("Decision Tree saved.")
