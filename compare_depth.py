import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
data = pd.read_csv("data/processed/cleaned_student_data.csv")

features = [
    "Age", "Study_Hours", "Attendance",
    "Previous_Score", "Quiz_Score",
    "Study_Frequency", "Interest",
    "Learning_Style", "Preferred_Difficulty",
    "Completed_Topics", "Weak_Topic",
    "Preferred_Resource"
]

numerical_features = [
    "Age", "Study_Hours", "Attendance",
    "Previous_Score", "Quiz_Score",
    "Study_Frequency", "Completed_Topics"
]

categorical_features = [
    "Interest", "Learning_Style",
    "Preferred_Difficulty", "Weak_Topic",
    "Preferred_Resource"
]

X = data[features]
y = data["Recommended_Topic"]

# Same training and testing split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDECISION TREE DEPTH COMPARISON")
print("-" * 75)
print(f"{'Depth':<12}{'Accuracy':<15}{'Precision':<15}{'Recall':<15}{'F1 Score'}")
print("-" * 75)

for depth in [5, 8, 10, 12, None]:

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=42
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions) * 100
    precision = precision_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    ) * 100
    recall = recall_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    ) * 100
    f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    ) * 100

    print(
        f"{str(depth):<12}"
        f"{accuracy:<15.2f}"
        f"{precision:<15.2f}"
        f"{recall:<15.2f}"
        f"{f1:.2f}"
    )

print("-" * 75)