class Student:
    """Store information about a student."""

    def __init__(
        self,
        age,
        study_hours,
        attendance,
        previous_score,
        quiz_score,
        study_frequency,
        interest,
        learning_style,
        preferred_difficulty,
        completed_topics
    ):
        self.age = age
        self.study_hours = study_hours
        self.attendance = attendance
        self.previous_score = previous_score
        self.quiz_score = quiz_score
        self.study_frequency = study_frequency
        self.interest = interest
        self.learning_style = learning_style
        self.preferred_difficulty = preferred_difficulty
        self.completed_topics = completed_topics

    def to_dict(self):
        """Convert student information into a dictionary."""

        return {
            "Age": self.age,
            "Study_Hours": self.study_hours,
            "Attendance": self.attendance,
            "Previous_Score": self.previous_score,
            "Quiz_Score": self.quiz_score,
            "Study_Frequency": self.study_frequency,
            "Interest": self.interest,
            "Learning_Style": self.learning_style,
            "Preferred_Difficulty": self.preferred_difficulty,
            "Completed_Topics": self.completed_topics,
            "Preferred_Resource": self.get_resource_preference()
        }

    def get_resource_preference(self):
        """Return resource type based on learning style."""

        preferences = {
            "Visual": "Video",
            "Reading": "Notes",
            "Practical": "Exercises",
            "Audio": "Audio"
        }

        return preferences.get(
            self.learning_style,
            "Notes"
        )

    def display_profile(self):
        """Display the student's profile."""

        print("\nStudent Profile:")
        print(f"Age: {self.age}")
        print(f"Interest: {self.interest}")
        print(f"Study Hours: {self.study_hours}")
        print(f"Attendance: {self.attendance}%")
        print(f"Previous Score: {self.previous_score}")
        print(f"Quiz Score: {self.quiz_score}")
        print(f"Learning Style: {self.learning_style}")
        print(f"Preferred Difficulty: {self.preferred_difficulty}")
        print(f"Completed Topics: {self.completed_topics}")