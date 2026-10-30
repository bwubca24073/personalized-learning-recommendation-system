# Personalized Learning Recommendation System

A Python-based machine learning system that recommends suitable learning topics and learning resources to students based on their learning profile and academic behaviour.

---

## 📌 Project Overview

The **Personalized Learning Recommendation System** is a machine-learning-based academic project designed to provide personalized learning recommendations to students.

The system collects information such as:

* Study hours
* Attendance
* Previous academic score
* Quiz score
* Study frequency
* Area of interest
* Learning style
* Preferred difficulty
* Number of completed topics

Using these inputs, the system predicts a suitable learning topic and recommends learning resources that match the student's learning preferences.

The project uses a **Decision Tree Classifier** for topic recommendation.

---

## 🎯 Objectives

The main objectives of the project are:

1. Collect and prepare student learning data.
2. Clean and preprocess the dataset.
3. Perform Exploratory Data Analysis (EDA).
4. Identify important features related to learning recommendations.
5. Develop a machine learning recommendation model.
6. Predict a suitable learning topic for a student.
7. Match the predicted topic with suitable learning resources.
8. Evaluate the machine learning model.
9. Visualize important patterns and model results.
10. Provide an easy-to-use console-based recommendation system.

---

## ✨ Features

* Student profile input
* Automatic data preprocessing
* Exploratory Data Analysis
* Feature selection
* Decision Tree classification
* Personalized topic recommendation
* Learning resource matching
* Learning-style-based resource matching
* Difficulty-based resource matching
* Resource ranking
* Recommendation explanation
* Model evaluation
* Feature importance analysis
* Confusion matrix visualization
* Interactive command-line menu

---

## 🔄 System Workflow

```text
Student Learning Dataset
          ↓
Data Cleaning
          ↓
Data Preprocessing
          ↓
Exploratory Data Analysis
          ↓
Feature Selection
          ↓
Decision Tree Model
          ↓
Student Input
          ↓
Predicted Learning Topic
          ↓
Resource Matching
          ↓
Resource Ranking
          ↓
Personalized Recommendation
          ↓
Evaluation & Visualization
```

---

## 🧠 Machine Learning Approach

### Algorithm Used

**Decision Tree Classifier**

A Decision Tree was selected because:

* It is easy to understand.
* It works well for classification problems.
* Its decisions can be interpreted.
* It works with both numerical and categorical features after preprocessing.
* Feature importance can be calculated.
* It is appropriate for a moderate-level academic project.

### Model Configuration

```text
Algorithm: Decision Tree Classifier
Maximum Depth: 8
Random State: 42
Training Data: 80%
Testing Data: 20%
```

---

## 📊 Dataset

The project uses a **synthetically generated dataset containing 500 student records**.

### Dataset Features

| Feature              | Description                       |
| -------------------- | --------------------------------- |
| Student_ID           | Unique student identifier         |
| Age                  | Student age                       |
| Study_Hours          | Daily study hours                 |
| Attendance           | Attendance percentage             |
| Previous_Score       | Previous academic score           |
| Quiz_Score           | Quiz performance                  |
| Study_Frequency      | Study sessions/frequency per week |
| Interest             | Main learning interest            |
| Learning_Style       | Preferred learning style          |
| Preferred_Difficulty | Preferred difficulty level        |
| Completed_Topics     | Number of completed topics        |
| Weak_Topic           | Student's weak topic              |
| Preferred_Resource   | Preferred resource type           |
| Recommended_Topic    | Target learning topic             |

### Learning Interests

* Programming
* Data Science
* Web Development
* Mathematics

### Learning Styles

* Visual
* Reading
* Practical
* Audio

### Difficulty Levels

* Easy
* Medium
* Hard

---

## 🧹 Data Preprocessing

The preprocessing stage performs:

* Missing-value detection
* Duplicate detection
* Numeric data conversion
* Range validation
* Missing categorical-value handling
* Removal of invalid numeric records
* Saving of cleaned data

The cleaned dataset is stored at:

```text
data/processed/cleaned_student_data.csv
```

---

## 🔎 Feature Selection

The following features are used by the machine learning model:

```text
Age
Study_Hours
Attendance
Previous_Score
Quiz_Score
Study_Frequency
Interest
Learning_Style
Preferred_Difficulty
Completed_Topics
Preferred_Resource
```

The following fields are excluded:

### Student_ID

It is only an identifier and does not provide useful learning information.

### Weak_Topic

It is excluded from the model input to avoid potential target leakage and to make the recommendation depend on the student's broader learning profile.

### Recommended_Topic

This is the target variable that the model predicts.

---

## 📈 Exploratory Data Analysis

The project generates several visualizations:

```text
interest_distribution.png
learning_style_distribution.png
study_hours_vs_score.png
attendance_vs_score.png
recommendation_distribution.png
```

These visualizations help understand:

* Distribution of student interests
* Distribution of learning styles
* Relationship between study hours and scores
* Relationship between attendance and scores
* Distribution of recommended learning topics

---

## 🤖 Recommendation System

The recommendation process works in several stages.

### Step 1 — Student Input

The user enters:

```text
Age
Study Hours
Attendance
Previous Score
Quiz Score
Study Frequency
Interest
Learning Style
Preferred Difficulty
Completed Topics
```

### Step 2 — ML Prediction

The Decision Tree model predicts a suitable learning topic.

Example:

```text
Interest: Programming
Study Hours: 3.5
Previous Score: 73
Quiz Score: 82
Preferred Difficulty: Medium

Predicted Topic:
Python Functions
```

### Step 3 — Resource Matching

The system searches the learning-resource dataset for resources related to the predicted topic.

Resources are ranked according to:

* Learning-style compatibility
* Difficulty compatibility

### Match Score

```text
Learning-style match  → +2
Difficulty match      → +3
```

Therefore, a resource matching both receives:

```text
Match Score = 5
```

Resources are then sorted from highest to lowest match score.

---

## 🧑‍💻 Object-Oriented Programming

The project includes a `Student` class.

The class stores student information and provides methods for:

* Storing student attributes
* Converting student information into dictionary format
* Determining preferred resource type
* Displaying the student profile

This demonstrates the use of:

* Classes
* Objects
* Constructors
* Methods
* Dictionaries

---

## 📏 Model Evaluation

The dataset is divided into:

```text
80% Training
20% Testing
```

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Current Test Results

```text
Test Samples: 100

Accuracy : 98.00%
Precision: 98.67%
Recall   : 98.00%
F1-Score : 98.15%
```

> These results are test-set results obtained from the synthetic dataset used in this academic project.

---

## 🔍 Feature Importance

The Decision Tree provides feature importance values.

Current feature importance results include:

```text
Study_Hours              0.4531
Interest                 0.3892
Preferred_Resource       0.0706
Quiz_Score               0.0505
Previous_Score           0.0263
Preferred_Difficulty     0.0054
Study_Frequency          0.0049
Attendance               0.0000
Age                      0.0000
Completed_Topics         0.0000
Learning_Style           0.0000
```

This indicates that, within the current synthetic dataset and trained Decision Tree, **Study_Hours and Interest contribute most strongly to the model's predictions**.

Feature importance should be interpreted as model-specific information rather than a general statement about real-world student learning.

The graph is saved as:

```text
visualizations/feature_importance.png
```

---

## 🖥️ Application Menu

The application provides the following menu:

```text
1. Get Personalized Recommendation
2. View Dataset Summary
3. View Model Performance
4. View EDA Graphs
5. View Feature Importance
6. Exit
```

---

## 📁 Project Structure

```text
Personalized Learning Recommendation System/
│
├── data/
│   ├── raw/
│   │   ├── student_learning_data.csv
│   │   └── learning_resources.csv
│   │
│   └── processed/
│       └── cleaned_student_data.csv
│
├── models/
│   └── learning_recommender.pkl
│
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── feature_selection.py
│   ├── generate_dataset.py
│   ├── preprocessing.py
│   ├── recommendation_model.py
│   ├── recommender.py
│   └── student.py
│
├── visualizations/
│   ├── interest_distribution.png
│   ├── learning_style_distribution.png
│   ├── study_hours_vs_score.png
│   ├── attendance_vs_score.png
│   ├── recommendation_distribution.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Joblib

### Development Environment

* Visual Studio Code
* Python Virtual Environment

---

## 🚀 Installation

### 1. Clone or download the project

Open the project folder in Visual Studio Code.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

From the project root directory:

```bash
python main.py
```

The interactive menu will appear.

---

## 🧪 Generating the Dataset

If the synthetic dataset needs to be regenerated:

```bash
python src/generate_dataset.py
```

This generates the student learning dataset used by the project.

---

## 🧹 Running Preprocessing

```bash
python src/preprocessing.py
```

This cleans the dataset and saves the processed version.

---

## 📊 Running EDA

```bash
python src/eda.py
```

This generates the EDA visualizations inside:

```text
visualizations/
```

---

## 🤖 Training the Model

```bash
python src/recommendation_model.py
```

The trained model is saved as:

```text
models/learning_recommender.pkl
```

---

## 📏 Running Evaluation

```bash
python src/evaluation.py
```

This calculates:

* Accuracy
* Precision
* Recall
* F1-Score

and generates the confusion matrix.

---

## ⚠️ Limitations

1. The current dataset is synthetic rather than collected from real students.
2. The recommendation target was generated using predefined rules.
3. Therefore, the high model performance may not represent real-world recommendation performance.
4. The current system is console-based and does not provide a graphical user interface.
5. The resource database is relatively small.
6. Real student feedback is not currently used to improve recommendations.
7. The system does not currently learn continuously from student interactions.

---

## 🔮 Future Improvements

Future versions could include:

* Real student learning datasets
* Larger learning-resource databases
* Student feedback collection
* Recommendation history
* Collaborative filtering
* More machine learning algorithms
* Automatic model comparison
* Web-based interface
* Student progress tracking
* Adaptive recommendations based on previous interactions
* Integration with online learning resources

---

## 🎓 Academic Concepts Demonstrated

This project demonstrates several Python and machine-learning concepts:

### Python

* Variables
* Conditions
* Loops
* Lists
* Dictionaries
* Functions
* Sorting
* File handling
* Exception handling
* Classes and objects

### Data Science

* NumPy
* Pandas
* Data cleaning
* Data preprocessing
* Exploratory Data Analysis
* Data visualization

### Machine Learning

* Feature selection
* Training/testing split
* Classification
* Decision Tree
* Model evaluation
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Feature importance

### Model Persistence

* Joblib

---

## 📝 Conclusion

The Personalized Learning Recommendation System demonstrates how machine learning can be used to provide topic recommendations based on student learning profiles and academic behaviour.

The system successfully performs data preprocessing, exploratory analysis, feature preparation, machine learning classification, personalized recommendation, resource matching, model evaluation, and visualization.

The current implementation provides a practical academic demonstration of a personalized learning recommendation workflow while leaving room for future improvement using real student data and larger resource collections.

---

## 👨‍💻 Project

**Personalized Learning Recommendation System**

Developed as an academic machine-learning project using Python and Scikit-learn.
