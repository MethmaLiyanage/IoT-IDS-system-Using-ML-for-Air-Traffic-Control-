import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def evaluate_ids(model, X_test, y_test, model_name, results_dir="results"):

    y_pred = model.predict(X_test)

    ATTACK_NAMES = ["Normal", "Backdoor", "Password", "DDoS", "Injection", "Ransomware", "XSS", "Scanning"]
    report = classification_report(
        y_test, y_pred,
        labels=list(range(8)),
        target_names=ATTACK_NAMES,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    print(f"\n--- {model_name} Report ---")
    print(report_df.round(3).to_string())

    report_df.to_csv(os.path.join(results_dir, f"{model_name}_report.csv"))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    SHORT = ["Norm", "Back", "Pass", "DDoS", "Inj", "Rans", "XSS", "Scan"]
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=SHORT,
                yticklabels=SHORT)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(os.path.join(results_dir, f"{model_name}_matrix.png"))
    plt.close()
