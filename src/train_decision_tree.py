from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier(
        max_depth=20,
        class_weight='balanced',
        random_state=42
    )

    dt.fit(X_train, y_train)
    return dt