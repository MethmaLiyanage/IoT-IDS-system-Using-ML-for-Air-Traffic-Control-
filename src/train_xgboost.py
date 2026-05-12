from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    return xgb
