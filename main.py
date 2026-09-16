import os

from src.recommender import get_recommendation
from src.evaluation import evaluate_model


DATA_FILE = "data/processed/cleaned_student_data.csv"


def show_dataset_summary():
    """Display basic information about the dataset."""

    try:
        import pandas as pd

        data = pd.read_csv(DATA_FILE)

        print("\n===============================================")
        print("              DATASET SUMMARY")
        print("===============================================")

        print(f"\nRows    : {data.shape[0]}")
        print(f"Columns : {data.shape[1]}")

        print("\nColumns:")

        for column in data.columns:
            print(f"- {column}")

        print("\nMissing Values:")
        print(data.isnull().sum())

        print("\nDataset Preview:")
        print(data.head())

    except FileNotFoundError:
        print("Error: Dataset not found.")

    except Exception as error:
        print(f"Error: {error}")


def show_model_performance():
    """Display model evaluation results."""

    print("\n===============================================")
    print("             MODEL PERFORMANCE")
    print("===============================================")

    print("\nModel:")
    print("Decision Tree Classifier")

    print("\nTraining/Test Split:")
    print("80% Training")
    print("20% Testing")

    evaluate_model()


def show_eda_graphs():
    """Display available EDA graph files."""

    print("\n===============================================")
    print("                EDA GRAPHS")
    print("===============================================")

    graph_folder = "visualizations"

    if not os.path.exists(graph_folder):
        print("Visualization folder not found.")
        return

    graphs = [
        "interest_distribution.png",
        "learning_style_distribution.png",
        "study_hours_vs_score.png",
        "attendance_vs_score.png",
        "recommendation_distribution.png",
        "confusion_matrix.png",
        "feature_importance.png"
    ]

    found = False

    for graph in graphs:

        path = os.path.join(
            graph_folder,
            graph
        )

        if os.path.exists(path):
            print(f"✓ {graph}")
            found = True
        else:
            print(f"✗ {graph}")

    if not found:
        print("\nNo graphs found.")




def show_feature_importance():
    """Calculate and display grouped feature importance."""

    try:
        import os
        import pandas as pd
        import joblib
        import matplotlib.pyplot as plt

        MODEL_FILE = "models/learning_recommender.pkl"

        # Load trained model
        model = joblib.load(MODEL_FILE)

        # Get pipeline components
        preprocessor = model.named_steps["preprocessor"]
        decision_tree = model.named_steps["model"]

        # Get transformed feature names
        feature_names = preprocessor.get_feature_names_out()

        # Get feature importance values
        importances = decision_tree.feature_importances_

        # Create DataFrame
        importance_data = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances
            }
        )

        # Original input features
        original_features = [
            "Age",
            "Study_Hours",
            "Attendance",
            "Previous_Score",
            "Quiz_Score",
            "Study_Frequency",
            "Completed_Topics",
            "Interest",
            "Learning_Style",
            "Preferred_Difficulty",
            "Preferred_Resource"
        ]

        grouped_importance = []

        # Group encoded features back into original features
        for feature in original_features:

            total_importance = 0.0

            for _, row in importance_data.iterrows():

                encoded_name = str(row["Feature"])

                # Remove everything before "__"
                if "__" in encoded_name:
                    clean_name = encoded_name.split(
                        "__",
                        1
                    )[1]
                else:
                    clean_name = encoded_name

                # Exact numerical feature
                if clean_name == feature:

                    total_importance += row["Importance"]

                # One-hot encoded categorical feature
                elif clean_name.startswith(
                    feature + "_"
                ):

                    total_importance += row["Importance"]

            grouped_importance.append(
                {
                    "Feature": feature,
                    "Importance": total_importance
                }
            )

        # Convert to DataFrame
        grouped_data = pd.DataFrame(
            grouped_importance
        )

        # Sort from highest to lowest
        grouped_data = grouped_data.sort_values(
            by="Importance",
            ascending=False
        )

        print("\n===============================================")
        print("        FEATURE IMPORTANCE ANALYSIS")
        print("===============================================\n")

        print("Features ranked by importance:\n")

        for _, row in grouped_data.iterrows():

            print(
                f"{row['Feature']:<25}"
                f"{row['Importance']:.4f}"
            )

        # Select top 10 features
        top_features = grouped_data.head(10)

        # Create graph
        plt.figure(figsize=(10, 6))

        plt.barh(
            top_features["Feature"][::-1],
            top_features["Importance"][::-1]
        )

        plt.xlabel("Importance")
        plt.ylabel("Feature")

        plt.title(
            "Feature Importance for Learning Recommendation"
        )

        plt.tight_layout()

        # Create visualization folder
        os.makedirs(
            "visualizations",
            exist_ok=True
        )

        output_file = (
            "visualizations/feature_importance.png"
        )

        plt.savefig(output_file)

        plt.close()

        print(
            "\nFeature importance graph saved to:"
            f"\n{output_file}"
        )

    except FileNotFoundError:

        print(
            "Error: Model file not found."
        )

    except Exception as error:

        print(
            "Error calculating feature importance:"
            f" {error}"
        )



def main():

    while True:

        print("\n\n===============================================")
        print(" PERSONALIZED LEARNING RECOMMENDATION SYSTEM")
        print("===============================================")

        print("\n1. Get Personalized Recommendation")
        print("2. View Dataset Summary")
        print("3. View Model Performance")
        print("4. View EDA Graphs")
        print("5. View Feature Importance")
        print("6. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":

            get_recommendation()

        elif choice == "2":

            show_dataset_summary()

        elif choice == "3":

            show_model_performance()

        elif choice == "4":

            show_eda_graphs()

        elif choice == "5":

            show_feature_importance()

        elif choice == "6":

            print("\nThank you for using the system! 👋")
            break

        else:

            print(
                "\nInvalid choice. "
                "Please enter a number from 1 to 6."
            )


if __name__ == "__main__":
    main()