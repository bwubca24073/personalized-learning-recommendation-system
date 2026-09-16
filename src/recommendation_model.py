import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_FILE = "data/processed/cleaned_student_data.csv"
MODEL_FILE = "models/learning_recommender.pkl"


def train_model():
    """Train and save the Decision Tree recommendation model."""

    try:
        data = pd.read_csv(DATA_FILE)

    except FileNotFoundError:
        print("Error: Cleaned dataset not found.")
        return

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
        "Preferred_Resource"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n======================================")
    print("        MODEL TRAINING")
    print("======================================")

    print(f"\nTraining Samples: {len(X_train)}")
    print(f"Testing Samples : {len(X_test)}")
    print(f"Accuracy        : {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    joblib.dump(
        pipeline,
        MODEL_FILE
    )

    print("\nModel saved successfully:")
    print(MODEL_FILE)


if __name__ == "__main__":
    train_model()

