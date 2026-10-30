import pandas as pd


INPUT_FILE = "data/processed/cleaned_student_data.csv"


def load_data(file_path):
    """Load the cleaned dataset."""

    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return data

    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None

    except Exception as error:
        print(f"Error loading dataset: {error}")
        return None


def select_features(data):
    """Select features for the machine learning model."""

    target = "Recommended_Topic"

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

    X = data[features]
    y = data[target]

    return X, y


def main():

    print("\n======================================")
    print("       FEATURE SELECTION")
    print("======================================")

    data = load_data(INPUT_FILE)

    if data is None:
        return

    X, y = select_features(data)

    print("\nSelected Features:")

    for feature in X.columns:
        print("-", feature)

    print("\nTarget:")
    print("-", y.name)

    print("\nNumber of input features:", len(X.columns))
    print("Number of records:", len(X))

    print("\nTarget Classes:")
    for topic in sorted(y.unique()):
        print("-", topic)

    print("\nFeature data preview:")
    print(X.head())

    print("\nTarget preview:")
    print(y.head())

    print("\nFeature selection completed successfully!")


if __name__ == "__main__":
    main()