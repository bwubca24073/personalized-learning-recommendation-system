import pandas as pd


def load_data(file_path):
    """Load student dataset from CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return data

    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None

    except Exception as error:
        print(f"Error while loading dataset: {error}")
        return None


def clean_data(data):
    """Clean and validate the student dataset."""

    print("\n--- Data Cleaning Started ---")

    # Show missing values
    print("\nMissing values before cleaning:")
    print(data.isnull().sum())

    # Remove duplicate records
    duplicate_count = data.duplicated().sum()
    print(f"\nDuplicate records found: {duplicate_count}")

    if duplicate_count > 0:
        data = data.drop_duplicates()

    # Check numeric columns
    numeric_columns = [
        "Age",
        "Study_Hours",
        "Attendance",
        "Previous_Score",
        "Quiz_Score",
        "Study_Frequency",
        "Completed_Topics"
    ]

    # Convert numeric columns to numbers
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Remove invalid numeric records
    data = data.dropna(subset=numeric_columns)

    # Validate ranges
    data = data[
        (data["Age"] > 0) &
        (data["Study_Hours"] >= 0) &
        (data["Attendance"].between(0, 100)) &
        (data["Previous_Score"].between(0, 100)) &
        (data["Quiz_Score"].between(0, 100)) &
        (data["Study_Frequency"] >= 0) &
        (data["Completed_Topics"] >= 0)
    ]

    # Fill missing categorical values
    categorical_columns = [
        "Interest",
        "Learning_Style",
        "Preferred_Difficulty",
        "Weak_Topic",
        "Preferred_Resource",
        "Recommended_Topic"
    ]

    for column in categorical_columns:
        if data[column].isnull().any():
            data[column] = data[column].fillna("Unknown")

    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    print("\nDataset shape after cleaning:")
    print(data.shape)

    print("\n--- Data Cleaning Completed ---")

    return data


def save_cleaned_data(data, output_path):
    """Save cleaned data to CSV file."""
    data.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")


if __name__ == "__main__":

    input_file = "data/raw/student_learning_data.csv"
    output_file = "data/processed/cleaned_student_data.csv"

    student_data = load_data(input_file)

    if student_data is not None:
        cleaned_data = clean_data(student_data)
        save_cleaned_data(cleaned_data, output_file)