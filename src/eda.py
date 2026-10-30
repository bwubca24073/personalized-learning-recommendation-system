import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "data/processed/cleaned_student_data.csv"
OUTPUT_DIR = "visualizations"
METRICS_FILE = "models/model_evaluation.csv"


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


def save_graph(file_name):
    """Save graph in the visualization folder."""

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, file_name),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {file_name}")


def plot_interest_distribution(data):
    """Display student interest distribution."""

    counts = data["Interest"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Student Interest Distribution")
    plt.xlabel("Interest")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=0)

    save_graph("interest_distribution.png")


def plot_learning_style_distribution(data):
    """Display learning style distribution."""

    counts = data["Learning_Style"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Learning Style Distribution")
    plt.xlabel("Learning Style")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=0)

    save_graph("learning_style_distribution.png")


def plot_study_hours_vs_score(data):
    """Show relationship between study hours and scores."""

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data["Study_Hours"],
        data["Previous_Score"],
        alpha=0.7
    )

    plt.title("Study Hours vs Previous Score")
    plt.xlabel("Study Hours")
    plt.ylabel("Previous Score")

    save_graph("study_hours_vs_score.png")


def plot_attendance_vs_score(data):
    """Show relationship between attendance and scores."""

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data["Attendance"],
        data["Previous_Score"],
        alpha=0.7
    )

    plt.title("Attendance vs Previous Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Previous Score")

    save_graph("attendance_vs_score.png")


def plot_recommendation_distribution(data):
    """Display recommended topic distribution."""

    counts = data["Recommended_Topic"].value_counts()

    plt.figure(figsize=(11, 6))
    counts.plot(kind="bar")

    plt.title("Recommended Topic Distribution")
    plt.xlabel("Recommended Topic")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=45, ha="right")

    save_graph("recommendation_distribution.png")


def plot_recommendation_by_interest(data):
    """Compare recommended topics across student interests."""

    cross_data = pd.crosstab(
        data["Interest"],
        data["Recommended_Topic"]
    )

    cross_data.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Recommended Topics by Student Interest")
    plt.xlabel("Student Interest")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=0)

    plt.legend(
        title="Recommended Topic",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    save_graph("recommendation_by_interest.png")


def plot_average_score_by_topic(data):
    """Compare average scores for recommended topics."""

    average_scores = data.groupby(
        "Recommended_Topic"
    )["Previous_Score"].mean().sort_values()

    plt.figure(figsize=(11, 6))
    average_scores.plot(kind="bar")

    plt.title("Average Previous Score by Recommended Topic")
    plt.xlabel("Recommended Topic")
    plt.ylabel("Average Previous Score")
    plt.xticks(rotation=45, ha="right")

    save_graph("average_score_by_topic.png")


def plot_study_hours_by_topic(data):
    """Compare average study hours for each recommended topic."""

    average_hours = data.groupby(
        "Recommended_Topic"
    )["Study_Hours"].mean().sort_values()

    plt.figure(figsize=(11, 6))
    average_hours.plot(kind="bar")

    plt.title("Average Study Hours by Recommended Topic")
    plt.xlabel("Recommended Topic")
    plt.ylabel("Average Study Hours")
    plt.xticks(rotation=45, ha="right")

    save_graph("study_hours_by_topic.png")


def plot_weak_topic_distribution(data):
    """Display the most common weak topics."""

    counts = data["Weak_Topic"].value_counts()

    plt.figure(figsize=(11, 6))
    counts.plot(kind="bar")

    plt.title("Student Weak Topic Distribution")
    plt.xlabel("Weak Topic")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=45, ha="right")

    save_graph("weak_topic_distribution.png")


def plot_model_performance():
    """Display Decision Tree model performance metrics."""

    if not os.path.exists(METRICS_FILE):
        print("\nModel evaluation file not found.")
        print("Please train the model first.")
        return

    try:
        metrics = pd.read_csv(METRICS_FILE)

        required_columns = {"Metric", "Score"}

        if not required_columns.issubset(metrics.columns):
            print("Invalid model evaluation file.")
            return

        plt.figure(figsize=(9, 5))

        bars = plt.bar(
            metrics["Metric"],
            metrics["Score"]
        )

        plt.title("Decision Tree Model Performance")
        plt.xlabel("Evaluation Metrics")
        plt.ylabel("Score (%)")
        plt.ylim(0, 105)

        for bar, value in zip(bars, metrics["Score"]):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.2f}%",
                ha="center"
            )

        save_graph("model_performance.png")

    except Exception as error:
        print(f"Error generating model performance graph: {error}")


def main():
    """Run the complete EDA process."""

    print("\n======================================")
    print("   EXPLORATORY DATA ANALYSIS")
    print("======================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_data(INPUT_FILE)

    if data is None:
        return

    show_dataset_summary(data)

    print("\nGenerating visualizations...\n")

    plot_interest_distribution(data)
    plot_learning_style_distribution(data)
    plot_study_hours_vs_score(data)
    plot_attendance_vs_score(data)
    plot_recommendation_distribution(data)
    plot_recommendation_by_interest(data)
    plot_average_score_by_topic(data)
    plot_study_hours_by_topic(data)
    plot_weak_topic_distribution(data)
    plot_model_performance()

    print("\nEDA completed successfully!")
    print("\nTotal graphs generated: 10")
    print("Graphs saved in: visualizations/")


if __name__ == "__main__":
    main()