from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from task_3.vectorization import get_vector_datasets


def try_classical_models():
    # Get data
    train_df, test_df = get_vector_datasets()

    # Prepare features and labels
    X_train = train_df.drop("sentiment", axis=1).values
    y_train = train_df["sentiment"].values
    X_test = test_df.drop("sentiment", axis=1).values
    y_test = test_df["sentiment"].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models to try
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(random_state=42),
    }

    # Try each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)

        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        # Print results
        print(f"{name} Results:")
        print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
        print("\nDetailed Test Report:")
        print(classification_report(y_test, test_pred))

        # For Random Forest and Gradient Boosting, print feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_features = sorted(
                zip(range(len(importances)), importances),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            print("\nTop 10 Most Important Features:")
            for idx, importance in top_features:
                print(f"Feature {idx}: {importance:.4f}")


if __name__ == "__main__":
    try_classical_models()
