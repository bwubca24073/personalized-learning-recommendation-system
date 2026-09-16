import random
import numpy as np
import pandas as pd


random.seed(42)
np.random.seed(42)


def generate_recommendation(
    interest,
    study_hours,
    previous_score,
    quiz_score,
    learning_style,
    difficulty
):
    """Generate a recommendation based on multiple student factors."""

    if interest == "Programming":

        if previous_score < 60 or quiz_score < 60:
            return "Python Basics"

        elif study_hours < 3:
            return "Python Functions"

        elif difficulty == "Hard" and study_hours >= 4:
            return "Python OOP"

        else:
            return "Python Functions"

    elif interest == "Data Science":

        if previous_score < 60 or quiz_score < 60:
            return "Statistics"

        elif study_hours >= 4.5 and difficulty == "Hard":
            return "Machine Learning"

        elif learning_style == "Practical":
            return "Pandas"

        else:
            return "NumPy"

    elif interest == "Web Development":

        if previous_score < 65:
            return "HTML CSS"

        elif study_hours >= 4:
            return "JavaScript"

        else:
            return "HTML CSS"

    else:

        if previous_score < 60 or quiz_score < 60:
            return "Algebra"

        elif study_hours >= 4:
            return "Probability"

        else:
            return "Statistics"


def generate_student_data(number_of_students=500):
    """Generate student learning dataset."""

    interests = [
        "Programming",
        "Data Science",
        "Web Development",
        "Mathematics"
    ]

    learning_styles = [
        "Visual",
        "Reading",
        "Practical",
        "Audio"
    ]

    resource_by_style = {
        "Visual": "Video",
        "Reading": "Notes",
        "Practical": "Exercises",
        "Audio": "Audio"
    }

    topic_data = {
        "Programming": [
            "Python Basics",
            "Python Functions",
            "Python OOP",
            "Machine Learning"
        ],
        "Data Science": [
            "Statistics",
            "Pandas",
            "NumPy",
            "Machine Learning"
        ],
        "Web Development": [
            "HTML CSS",
            "JavaScript"
        ],
        "Mathematics": [
            "Algebra",
            "Probability",
            "Statistics"
        ]
    }

    records = []

    for number in range(1, number_of_students + 1):

        age = random.randint(18, 24)

        study_hours = round(
            np.clip(
                np.random.normal(3.5, 1.5),
                0.5,
                8.0
            ),
            1
        )

        attendance = round(
            np.clip(
                np.random.normal(
                    75 + study_hours * 2,
                    12
                ),
                45,
                100
            ),
            1
        )

        previous_score = round(
            np.clip(
                30
                + study_hours * 7
                + attendance * 0.35
                + np.random.normal(0, 8),
                30,
                98
            ),
            1
        )

        quiz_score = round(
            np.clip(
                previous_score + np.random.normal(0, 6),
                25,
                100
            ),
            1
        )

        study_frequency = random.randint(1, 7)

        interest = random.choice(interests)

        learning_style = random.choice(learning_styles)

        if previous_score < 55:
            preferred_difficulty = "Easy"

        elif previous_score < 75:
            preferred_difficulty = "Medium"

        else:
            preferred_difficulty = "Hard"

        completed_topics = random.randint(2, 15)

        possible_topics = topic_data[interest]

        weak_topic = random.choice(possible_topics)

        preferred_resource = resource_by_style[learning_style]

        recommended_topic = generate_recommendation(
            interest,
            study_hours,
            previous_score,
            quiz_score,
            learning_style,
            preferred_difficulty
        )

        records.append({
            "Student_ID": f"S{number:03d}",
            "Age": age,
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Score": previous_score,
            "Quiz_Score": quiz_score,
            "Study_Frequency": study_frequency,
            "Interest": interest,
            "Learning_Style": learning_style,
            "Preferred_Difficulty": preferred_difficulty,
            "Completed_Topics": completed_topics,
            "Weak_Topic": weak_topic,
            "Preferred_Resource": preferred_resource,
            "Recommended_Topic": recommended_topic
        })

    return pd.DataFrame(records)


def save_dataset(data, file_path):
    """Save generated dataset."""

    data.to_csv(file_path, index=False)

    print(f"Dataset saved successfully to: {file_path}")


def main():

    print("Generating improved student learning dataset...")

    student_data = generate_student_data(500)

    print("\nDataset generated successfully!")
    print(f"Number of records: {len(student_data)}")
    print(f"Number of columns: {len(student_data.columns)}")

    print("\nRecommendation distribution:")

    print(
        student_data["Recommended_Topic"]
        .value_counts()
    )

    print("\nFirst 5 records:")
    print(student_data.head())

    save_dataset(
        student_data,
        "data/raw/student_learning_data.csv"
    )


if __name__ == "__main__":
    main()