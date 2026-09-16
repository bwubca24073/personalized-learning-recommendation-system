import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


DATA_PATH = "data/processed/cleaned_student_data.csv"


def load_data():
    """Load the cleaned student dataset."""
    return pd.read_csv(DATA_PATH)


def build_model():
    """Create the same Decision Tree model used by the project."""

    data = load_data()

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
        "Preferred_Resource"
    ]

    target = "Recommended_Topic"

    X = data[features]
    y = data[target]

    categorical_features = [
        "Interest",
        "Learning_Style",
        "Preferred_Difficulty",
        "Preferred_Resource"
    ]

    numerical_features = [
        "Age",
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Quiz_Score",
        "Study_Frequency",
        "Completed_Topics"
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

    model = DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return X, y, pipeline


def evaluate_model():
    """Train and evaluate the recommendation model."""

    X, y, pipeline = build_model()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n======================================")
    print("        MODEL EVALUATION")
    print("======================================")

    print(f"\nTest Samples: {len(y_test)}")

    print(f"\nAccuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")

    print("\nEvaluation completed successfully.")

    return pipeline, X_test, y_test, y_pred


def create_confusion_matrix(y_test, y_pred):
    """Create and save the confusion matrix."""

    matrix = confusion_matrix(y_test, y_pred)

    labels = sorted(y_test.unique())

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title("Recommendation Model Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "visualizations/confusion_matrix.png"
    )

    plt.close()

    print("\nConfusion matrix saved to:")
    print("visualizations/confusion_matrix.png")


if __name__ == "__main__":

    pipeline, X_test, y_test, y_pred = evaluate_model()

    create_confusion_matrix(
        y_test,
        y_pred
    )