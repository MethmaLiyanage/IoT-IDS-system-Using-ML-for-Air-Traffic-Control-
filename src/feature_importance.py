import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_importance(model, feature_names, top_n=15, results_dir="results"):

    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature importance.")
        return

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    feature_importance_df.to_csv(
        os.path.join(results_dir, "feature_importance_full.csv"), index=False
    )

    top_features = feature_importance_df.head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=top_features)
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "feature_importance_top.png"))
    plt.close()

    print(f"\nTop {top_n} Important Features:")
    print(top_features)
