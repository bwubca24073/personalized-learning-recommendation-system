import os
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
    ConfusionMatrixDisplay,
    classification_report
)


DATA_PATH = "data/processed/cleaned_student_data.csv"


def load_data():
    """Load the cleaned student dataset."""
    return pd.read_csv(DATA_PATH)


def build_model():
    """Create the Decision Tree recommendation model."""

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
        "Weak_Topic",
        "Preferred_Resource"
    ]

    target = "Recommended_Topic"

    X = data[features]
    y = data[target]

    categorical_features = [
        "Interest",
        "Learning_Style",
        "Preferred_Difficulty",
        "Weak_Topic",
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
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    return X, y, pipeline


def create_confusion_matrix(y_test, y_pred):
    """Create and save the confusion matrix."""

    labels = sorted(y_test.unique())

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    os.makedirs("visualizations", exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    display.plot(
        ax=axis,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False
    )

    plt.title("Recommendation Model Confusion Matrix")
    plt.tight_layout()

    output_path = "visualizations/confusion_matrix.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("\nConfusion matrix saved to:")
    print(output_path)


def show_class_performance(y_test, y_pred):
    """Display performance for every recommended topic."""

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_data = []

    for topic, scores in report.items():

        if topic in ["accuracy", "macro avg", "weighted avg"]:
            continue

        report_data.append({
            "Topic": topic,
            "Precision": scores["precision"],
            "Recall": scores["recall"],
            "F1-Score": scores["f1-score"],
            "Test Samples": scores["support"]
        })

    results = pd.DataFrame(report_data)

    results = results.sort_values(
        by="F1-Score",
        ascending=True
    )

    print("\n======================================")
    print("       TOPIC-WISE PERFORMANCE")
    print("======================================")

    print(
        results.to_string(
            index=False,
            formatters={
                "Precision": "{:.2f}".format,
                "Recall": "{:.2f}".format,
                "F1-Score": "{:.2f}".format
            }
        )
    )

    weak_topics = results[
        (results["F1-Score"] < 0.70) |
        (results["Test Samples"] < 5)
    ]

    print("\n======================================")
    print("       TOPIC QUALITY ANALYSIS")
    print("======================================")

    if weak_topics.empty:
        print("\nNo weak topics identified by the selected criteria.")
    else:
        print("\nTopics that need attention:")

        for _, row in weak_topics.iterrows():
            print(
                f"- {row['Topic']} "
                f"(F1-Score: {row['F1-Score']:.2f}, "
                f"Test Samples: {int(row['Test Samples'])})"
            )

    print(
        "\nNote: Topics with very few test samples "
        "need more data for reliable evaluation."
    )


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

    show_class_performance(y_test, y_pred)

    create_confusion_matrix(y_test, y_pred)

    print("\nEvaluation completed successfully.")

    return pipeline, X_test, y_test, y_pred


if __name__ == "__main__":

    evaluate_model()