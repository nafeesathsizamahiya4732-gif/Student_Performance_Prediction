# Student Performance Prediction

## Project Overview

This project predicts the grade class of students using machine learning techniques. The project uses student-related academic, demographic, and extracurricular information to predict the student's grade class.

## Dataset

The dataset contains information about 2392 students and 15 columns.

The features include:

- Age
- Gender
- Ethnicity
- Parental Education
- Study Time Weekly
- Absences
- Tutoring
- Parental Support
- Extracurricular Activities
- Sports
- Music
- Volunteering

The target variable is:

- GradeClass

## Data Preprocessing

The dataset was checked for:

- Missing values
- Duplicate records
- Dataset shape
- Column names
- Statistical information

StudentID was removed during preprocessing because it is only an identification value and does not contribute to prediction.

## Machine Learning Models

Four classification algorithms were trained:

1. Logistic Regression
2. Decision Tree
3. Naive Bayes
4. Support Vector Machine (SVM)

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 72.65% |
| Decision Tree | 58.04% |
| Naive Bayes | 67.22% |
| Support Vector Machine | 69.73% |

## Best Model

Logistic Regression achieved the highest accuracy of approximately **72.65%** among the four models tested.

## Sample Prediction

The trained Logistic Regression model was also used to predict the grade class of a sample student.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- PyCharm

## Project Files

- `main.py` - Main Python program
- `dataset/` - Dataset used for the project
- `student_performance_model.pkl` - Saved trained Logistic Regression model
- `README.md` - Project documentation