import pandas as pd
import joblib

from src.student import Student


MODEL_FILE = "models/learning_recommender.pkl"
RESOURCE_FILE = "data/raw/learning_resources.csv"


def load_model(file_path):
    """Load the trained ML model."""

    try:
        model = joblib.load(file_path)
        print("Model loaded successfully.")
        return model

    except FileNotFoundError:
        print("Error: Model file not found.")
        return None

    except Exception as error:
        print(f"Error loading model: {error}")
        return None


def load_resources(file_path):
    """Load learning resources."""

    try:
        resources = pd.read_csv(file_path)
        print("Learning resources loaded successfully.")
        return resources

    except FileNotFoundError:
        print("Error: Resource file not found.")
        return None

    except Exception as error:
        print(f"Error loading resources: {error}")
        return None


def get_number(prompt, minimum, maximum, data_type=float):
    """Get and validate numerical input."""

    while True:
        try:
            value = data_type(input(prompt))

            if minimum <= value <= maximum:
                return value

            print(
                f"Please enter a value between "
                f"{minimum} and {maximum}."
            )

        except ValueError:
            print("Invalid input. Please enter a number.")


def get_choice(prompt, options):
    """Get a valid menu choice."""

    while True:
        try:
            choice = int(input(prompt))

            if choice in options:
                return options[choice]

            print("Invalid choice. Please try again.")

        except ValueError:
            print("Please enter a valid number.")


def get_student_input():
    """Collect and validate student information."""

    print("\n======================================")
    print("       STUDENT INFORMATION")
    print("======================================")

    age = get_number(
        "Enter age (18-24): ",
        18,
        24,
        int
    )

    study_hours = get_number(
        "Enter daily study hours (0.5-8): ",
        0.5,
        8
    )

    attendance = get_number(
        "Enter attendance percentage (0-100): ",
        0,
        100
    )

    previous_score = get_number(
        "Enter previous score (0-100): ",
        0,
        100
    )

    quiz_score = get_number(
        "Enter quiz score (0-100): ",
        0,
        100
    )

    study_frequency = get_number(
        "Enter study frequency per week (1-7): ",
        1,
        7,
        int
    )

    print("\nInterest options:")

    interests = {
        1: "Programming",
        2: "Data Science",
        3: "Web Development",
        4: "Mathematics"
    }

    for number, interest in interests.items():
        print(f"{number}. {interest}")

    interest = get_choice(
        "Choose interest: ",
        interests
    )

    print("\nLearning Style options:")

    learning_styles = {
        1: "Visual",
        2: "Reading",
        3: "Practical",
        4: "Audio"
    }

    for number, style in learning_styles.items():
        print(f"{number}. {style}")

    learning_style = get_choice(
        "Choose learning style: ",
        learning_styles
    )

    print("\nDifficulty options:")

    difficulties = {
        1: "Easy",
        2: "Medium",
        3: "Hard"
    }

    for number, difficulty in difficulties.items():
        print(f"{number}. {difficulty}")

    preferred_difficulty = get_choice(
        "Choose preferred difficulty: ",
        difficulties
    )

    completed_topics = get_number(
        "Enter completed topics (0-50): ",
        0,
        50,
        int
    )

    # Create Student object
    student = Student(
        age=age,
        study_hours=study_hours,
        attendance=attendance,
        previous_score=previous_score,
        quiz_score=quiz_score,
        study_frequency=study_frequency,
        interest=interest,
        learning_style=learning_style,
        preferred_difficulty=preferred_difficulty,
        completed_topics=completed_topics
    )

    return student


def recommend_topic(model, student):
    """Predict the recommended learning topic."""

    student_data = student.to_dict()

    student_df = pd.DataFrame([student_data])

    prediction = model.predict(student_df)

    return prediction[0]


def explain_recommendation(student, topic):
    """Explain the student's recommendation."""

    print("\nWhy this topic was recommended:")

    print(
        f"✓ Your interest is {student.interest}."
    )

    print(
        f"✓ Your previous score is "
        f"{student.previous_score:.1f}."
    )

    print(
        f"✓ Your quiz score is "
        f"{student.quiz_score:.1f}."
    )

    print(
        f"✓ You study about "
        f"{student.study_hours:.1f} hours per day."
    )

    print(
        f"✓ Your preferred difficulty is "
        f"{student.preferred_difficulty}."
    )

    print(
        f"✓ Your learning style is "
        f"{student.learning_style}."
    )

    print(
        f"✓ The ML model predicted "
        f"'{topic}' based on your learning profile."
    )


def find_resources(
    resources,
    topic,
    learning_style,
    difficulty
):
    """Find and rank learning resources."""

    resource_type_map = {
        "Visual": "Video",
        "Reading": "Notes",
        "Practical": "Exercises",
        "Audio": "Audio"
    }

    resource_type = resource_type_map[learning_style]

    # Find resources for the recommended topic
    topic_resources = resources[
        resources["Topic"] == topic
    ].copy()

    if topic_resources.empty:
        return topic_resources, "No match"

    # Calculate matching score
    def calculate_score(resource):
        score = 0

        # Learning-style/resource-type match
        if resource["Resource_Type"] == resource_type:
            score += 2

        # Difficulty match
        if resource["Difficulty"] == difficulty:
            score += 3

        return score

    topic_resources["Match_Score"] = (
        topic_resources.apply(
            calculate_score,
            axis=1
        )
    )

    # Sort resources by highest matching score
    topic_resources = topic_resources.sort_values(
        by=["Match_Score", "Resource_ID"],
        ascending=[False, True]
    )

    # Determine match type
    best_score = topic_resources.iloc[0]["Match_Score"]

    if best_score == 5:
        match_type = "Exact match"
    elif best_score == 3:
        match_type = "Difficulty match"
    elif best_score == 2:
        match_type = "Learning-style match"
    else:
        match_type = "Topic match"

    return topic_resources, match_type


def display_recommendation(
    student,
    topic,
    resources,
    match_type
):
    """Display the final recommendation."""

    print("\n======================================")
    print("     PERSONALIZED RECOMMENDATION")
    print("======================================")

    student.display_profile()

    print("\n--------------------------------------")

    print("Recommended Topic:")
    print(f">>> {topic}")

    explain_recommendation(
        student,
        topic
    )

    print("\nRecommended Resources:")

    if resources.empty:
        print("No resources found for this topic.")
        return

    for _, resource in resources.iterrows():

        print("\nResource ID:", resource["Resource_ID"])
        print("Resource:", resource["Resource_Name"])
        print("Type:", resource["Resource_Type"])
        print("Difficulty:", resource["Difficulty"])
        print("Match Score:", resource["Match_Score"])

    print("\nMatch Type:", match_type)


def get_recommendation():
    """Run the complete personalized recommendation process."""

    print("\n======================================")
    print(" PERSONALIZED LEARNING RECOMMENDER")
    print("======================================")

    model = load_model(MODEL_FILE)

    if model is None:
        return

    resources = load_resources(RESOURCE_FILE)

    if resources is None:
        return

    student = get_student_input()

    topic = recommend_topic(
        model,
        student
    )

    recommended_resources, match_type = find_resources(
        resources,
        topic,
        student.learning_style,
        student.preferred_difficulty
    )

    display_recommendation(
        student,
        topic,
        recommended_resources,
        match_type
    )


def main():

    get_recommendation()


if __name__ == "__main__":
    main()
