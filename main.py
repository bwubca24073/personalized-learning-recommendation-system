import os
import glob
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.recommender import get_recommendation
from src.evaluation import evaluate_model


DATA_FILE = "data/processed/cleaned_student_data.csv"
MODEL_FILE = "models/learning_recommender.pkl"


def show_dataset_summary():
    """Display basic information about the dataset."""

    try:
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
    """Automatically find and display all available PNG graphs."""

    print("\n===============================================")
    print("                EDA GRAPHS")
    print("===============================================")

    graph_folder = "visualizations"

    if not os.path.exists(graph_folder):
        print("Visualization folder not found.")
        return

    graph_files = sorted(
        glob.glob(os.path.join(graph_folder, "*.png"))
    )

    if not graph_files:
        print("No graphs available.")
        return

    print(f"\nTotal graphs found: {len(graph_files)}")

    for graph in graph_files:
        print(f"✓ Found: {os.path.basename(graph)}")

    print("\nOpening graphs...")

    for path in graph_files:
        try:
            image = plt.imread(path)

            plt.figure(figsize=(9, 5))
            plt.imshow(image)
            plt.axis("off")
            plt.title(os.path.basename(path))
            plt.tight_layout()
            plt.show()
            plt.close()

        except Exception as error:
            print(f"Could not open {path}: {error}")


def show_feature_importance():
    """Calculate and display grouped feature importance."""

    try:
        model = joblib.load(MODEL_FILE)

        preprocessor = model.named_steps["preprocessor"]
        decision_tree = model.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()
        importances = decision_tree.feature_importances_

        importance_data = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

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
            "Weak_Topic",
            "Preferred_Resource"
        ]

        grouped_importance = []

        for feature in original_features:
            total_importance = 0.0

            for _, row in importance_data.iterrows():
                encoded_name = str(row["Feature"])

                if "__" in encoded_name:
                    clean_name = encoded_name.split("__", 1)[1]
                else:
                    clean_name = encoded_name

                if clean_name == feature:
                    total_importance += row["Importance"]

                elif clean_name.startswith(feature + "_"):
                    total_importance += row["Importance"]

            grouped_importance.append({
                "Feature": feature,
                "Importance": total_importance
            })

        grouped_data = pd.DataFrame(grouped_importance)

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

        top_features = grouped_data.head(10)

        plt.figure(figsize=(10, 6))

        plt.barh(
            top_features["Feature"][::-1],
            top_features["Importance"][::-1]
        )

        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance for Learning Recommendation")

        plt.tight_layout()

        os.makedirs("visualizations", exist_ok=True)

        output_file = "visualizations/feature_importance.png"
        plt.savefig(output_file)

        print(f"\nFeature importance graph saved to: {output_file}")

        plt.show()
        plt.close()

    except FileNotFoundError:
        print("Error: Model file not found.")

    except Exception as error:
        print(f"Error calculating feature importance: {error}")


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
            print("\nThank you for using the system!")
            break

        else:
            print("\nInvalid choice. Please enter a number from 1 to 6.")


if __name__ == "__main__":
    main()