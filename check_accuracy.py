import pandas as pd

models = ['Random_Forest', 'Decision_Tree', 'LSTM']

print("\n========== Model Accuracies ==========")
for name in models:
    try:
        df = pd.read_csv(f'src/results/{name}_report.csv', index_col=0)
        acc = float(df.loc['accuracy', 'f1-score']) * 100
        print(f"  {name:<20}: {acc:.2f}%")
    except Exception as e:
        print(f"  {name:<20}: Not found ({e})")
print("======================================\n")
