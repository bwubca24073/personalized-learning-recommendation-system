import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

DATA_FILE = "data/processed/cleaned_student_data.csv"
MODEL_FILE = "models/learning_recommender.pkl"


def train_model():

    print("\nLoading cleaned dataset...")

    data = pd.read_csv(DATA_FILE)

    features = [
        "Age",
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Quiz_Score",
        "Study_Frequency",
        "Interest",
        "Learning_Style",
        "Preferred_Difficulty",
        "Completed_Topics",
        "Weak_Topic",
        "Preferred_Resource"
    ]

    target = "Recommended_Topic"

    X = data[features]
    y = data[target]

    numerical_features = [
        "Age",
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Quiz_Score",
        "Study_Frequency",
        "Completed_Topics"
    ]

    categorical_features = [
        "Interest",
        "Learning_Style",
        "Preferred_Difficulty",
        "Weak_Topic",
        "Preferred_Resource"
    ]

    preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

    # Decision Tree Classifier
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=42
                )
            )
        ]
    )

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Decision Tree model...")

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    from sklearn.metrics import confusion_matrix
    print('\nCONFUSION MATRIX:\n', confusion_matrix(y_test, y_pred, labels=model.classes_))
    print('\nCLASS ORDER:\n', model.classes_)
    

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    # Display overall evaluation
    print("\n===================================")
    print("       MODEL EVALUATION")
    print("===================================")

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    # Display individual topic performance
    print("\n===================================")
    print("   TOPIC-WISE PERFORMANCE")
    print("===================================")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)

    # Save evaluation results
    os.makedirs("models", exist_ok=True)

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            round(accuracy * 100, 2),
            round(precision * 100, 2),
            round(recall * 100, 2),
            round(f1 * 100, 2)
        ]
    })

    metrics.to_csv(
        "models/model_evaluation.csv",
        index=False
    )

    # Save trained model
    joblib.dump(model, MODEL_FILE)

    print("\nModel and evaluation results saved successfully!")
    print(f"Model location: {MODEL_FILE}")
    print("Evaluation file: models/model_evaluation.csv")


if __name__ == "__main__":
    train_model()