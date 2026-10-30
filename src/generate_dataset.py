import random
import numpy as np
import pandas as pd


random.seed(42)
np.random.seed(42)


def generate_recommendation(
    interest,
    study_hours,
    attendance,
    previous_score,
    quiz_score,
    study_frequency,
    learning_style,
    difficulty,
    completed_topics,
    weak_topic
):
    """Recommend a topic using student performance and preferences."""

    average_score = (previous_score + quiz_score) / 2

    if interest == "Programming":

        if weak_topic == "Python Basics" or average_score < 55:
            return "Python Basics"

        elif weak_topic == "Python OOP":
            return "Python OOP"

        elif average_score >= 75 and study_hours >= 4:
            return "Python OOP"

        else:
            return "Python Functions"

    elif interest == "Data Science":

        if weak_topic == "Statistics" or average_score < 55:
            return "Statistics"

        elif weak_topic == "Pandas":
            return "Pandas"

        elif weak_topic == "Machine Learning":
            return "Machine Learning"

        elif weak_topic == "NumPy":
            return "NumPy"

        elif average_score >= 80 and study_hours >= 4.5:
            return "Machine Learning"

        elif learning_style == "Practical" and completed_topics >= 5:
            return "Pandas"

        else:
            return "NumPy"

    elif interest == "Web Development":

        if weak_topic == "HTML CSS" or average_score < 60:
            return "HTML CSS"

        elif weak_topic == "JavaScript":
            return "JavaScript"

        elif average_score >= 70 and completed_topics >= 5:
            return "JavaScript"

        else:
            return "HTML CSS"

    else:

        if weak_topic == "Algebra" or average_score < 55:
            return "Algebra"

        elif weak_topic == "Probability":
            return "Probability"

        elif average_score >= 70 and study_frequency >= 5:
            return "Probability"

        else:
            return "Statistics"


def generate_student_data(number_of_students=1000):

    weak_topics = {
        "Programming": [
            "Python Basics",
            "Python Functions",
            "Python OOP"
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
            "Probability"
        ]
    }

    learning_styles = ["Visual", "Reading", "Practical", "Audio"]

    resource_by_style = {
        "Visual": "Video",
        "Reading": "Notes",
        "Practical": "Exercises",
        "Audio": "Audio"
    }

    interests = list(weak_topics.keys())

    records = []

    for number in range(1, number_of_students + 1):

        # Distribute interests equally
        interest = interests[(number - 1) % len(interests)]

        # Distribute weak topics within each interest
        topic_list = weak_topics[interest]
        topic_index = (number - 1) // len(interests)
        weak_topic = topic_list[topic_index % len(topic_list)]

        age = random.randint(18, 24)

        study_hours = round(
            np.clip(np.random.normal(3.5, 1.5), 0.5, 8.0), 1
        )

        attendance = round(
            np.clip(
                np.random.normal(75 + study_hours * 2, 12),
                45, 100
            ), 1
        )

        previous_score = round(
            np.clip(
                30 + study_hours * 7 + attendance * 0.35
                + np.random.normal(0, 8),
                30, 98
            ), 1
        )

        quiz_score = round(
            np.clip(
                previous_score + np.random.normal(0, 6),
                25, 100
            ), 1
        )

        study_frequency = random.randint(1, 7)
        learning_style = random.choice(learning_styles)

        if previous_score < 55:
            preferred_difficulty = "Easy"
        elif previous_score < 75:
            preferred_difficulty = "Medium"
        else:
            preferred_difficulty = "Hard"

        completed_topics = random.randint(2, 15)
        preferred_resource = resource_by_style[learning_style]

        recommended_topic = generate_recommendation(
            interest,
            study_hours,
            attendance,
            previous_score,
            quiz_score,
            study_frequency,
            learning_style,
            preferred_difficulty,
            completed_topics,
            weak_topic
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

    data = pd.DataFrame(records)

    data = data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return data


def save_dataset(data, file_path):

    data.to_csv(file_path, index=False)

    print(f"\nDataset saved successfully to: {file_path}")


def main():

    print("Generating personalized student learning dataset...")

    student_data = generate_student_data(1000)

    print("\nDataset generated successfully!")
    print(f"Number of records: {len(student_data)}")
    print(f"Number of columns: {len(student_data.columns)}")

    print("\nRecommendation distribution:")
    print(
        student_data["Recommended_Topic"]
        .value_counts()
        .to_string()
    )

    print("\nFirst 5 records:")
    print(student_data.head())

    save_dataset(
        student_data,
        "data/raw/student_learning_data.csv"
    )


if __name__ == "__main__":
    main()