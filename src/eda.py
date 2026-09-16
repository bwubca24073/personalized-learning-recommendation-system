import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "data/processed/cleaned_student_data.csv"
OUTPUT_DIR = "visualizations"


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


def show_dataset_summary(data):
    """Display basic dataset information."""

    print("\n========== DATASET SUMMARY ==========")

    print(f"Number of students: {len(data)}")
    print(f"Number of features: {len(data.columns)}")

    print("\nColumns:")
    for column in data.columns:
        print("-", column)

    print("\nNumerical Statistics:")
    print(data.describe())


def plot_interest_distribution(data):
    """Create interest distribution chart."""

    counts = data["Interest"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Student Interest Distribution")
    plt.xlabel("Interest")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/interest_distribution.png")
    plt.close()


def plot_learning_style_distribution(data):
    """Create learning style distribution chart."""

    counts = data["Learning_Style"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Learning Style Distribution")
    plt.xlabel("Learning Style")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/learning_style_distribution.png")
    plt.close()


def plot_study_hours_vs_score(data):
    """Create study hours vs score scatter plot."""

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data["Study_Hours"],
        data["Previous_Score"]
    )

    plt.title("Study Hours vs Previous Score")
    plt.xlabel("Study Hours")
    plt.ylabel("Previous Score")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/study_hours_vs_score.png")
    plt.close()


def plot_attendance_vs_score(data):
    """Create attendance vs score scatter plot."""

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data["Attendance"],
        data["Previous_Score"]
    )

    plt.title("Attendance vs Previous Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Previous Score")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attendance_vs_score.png")
    plt.close()


def plot_recommendation_distribution(data):
    """Create recommended topic distribution chart."""

    counts = data["Recommended_Topic"].value_counts()

    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar")

    plt.title("Recommended Topic Distribution")
    plt.xlabel("Recommended Topic")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/recommendation_distribution.png")
    plt.close()


def main():
    """Run the complete EDA process."""

    print("\n======================================")
    print("   EXPLORATORY DATA ANALYSIS")
    print("======================================")

    data = load_data(INPUT_FILE)

    if data is None:
        return

    show_dataset_summary(data)

    print("\nGenerating visualizations...")

    plot_interest_distribution(data)
    plot_learning_style_distribution(data)
    plot_study_hours_vs_score(data)
    plot_attendance_vs_score(data)
    plot_recommendation_distribution(data)

    print("\nEDA completed successfully!")

    print("\nGraphs saved in:")
    print("visualizations/")


if __name__ == "__main__":
    main()