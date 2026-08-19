
# STUDENT PERFORMANCE PREDICTION SYSTEM




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset", "Student_performance_data.csv"))




print("\n========== FIRST 5 ROWS ==========")
print(df.head())


print("\n========== DATASET SHAPE ==========")
print("Number of rows and columns:", df.shape)


print("\n========== COLUMN NAMES ==========")
print(df.columns)


print("\n========== DATASET INFORMATION ==========")
df.info()


print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())



df = df.drop("StudentID", axis=1)


print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())


df = df.drop_duplicates()


print("\n========== DATASET AFTER PREPROCESSING ==========")
print("Rows and columns:", df.shape)

print("\n========== MISSING VALUES AFTER PREPROCESSING ==========")
print(df.isnull().sum())



X = df.drop(["GPA", "GradeClass"], axis=1)


y = df["GradeClass"]

print("\n========== FEATURES (X) ==========")
print(X.columns)

print("\n========== TARGET (y) ==========")
print(y.name)

print("\n========== FEATURE SHAPE ==========")
print(X.shape)

print("\n========== TARGET SHAPE ==========")
print(y.shape)


plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="StudyTimeWeekly",
    y="GPA"
)

plt.title("Study Time vs GPA")
plt.xlabel("Study Time (Weekly Hours)")
plt.ylabel("GPA")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Absences",
    y="GPA"
)

plt.title("Absences vs GPA")
plt.xlabel("Number of Absences")
plt.ylabel("GPA")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="GradeClass"
)

plt.title("Grade Class Distribution")
plt.xlabel("Grade Class")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



X = df[
    [
        "Age",
        "Gender",
        "Ethnicity",
        "ParentalEducation",
        "StudyTimeWeekly",
        "Absences",
        "Tutoring",
        "ParentalSupport",
        "Extracurricular",
        "Sports",
        "Music",
        "Volunteering"
    ]
]

y = df["GradeClass"]
y = df["GradeClass"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== DATA SPLIT ==========")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)




logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)




decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

decision_tree_model.fit(X_train, y_train)




naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GaussianNB())
])

naive_bayes_model.fit(X_train, y_train)



svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC())
])

svm_model.fit(X_train, y_train)


print("\n========== MODEL TRAINING COMPLETE ==========")
print("Logistic Regression: Trained")
print("Decision Tree: Trained")
print("Naive Bayes: Trained")
print("SVM: Trained")


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)




y_pred_logistic = logistic_model.predict(X_test)
y_pred_tree = decision_tree_model.predict(X_test)
y_pred_nb = naive_bayes_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
# Store model accuracies for comparison

logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
decision_tree_accuracy = accuracy_score(y_test, y_pred_tree)
naive_bayes_accuracy = accuracy_score(y_test, y_pred_nb)
svm_accuracy = accuracy_score(y_test, y_pred_svm)



print("\n========== LOGISTIC REGRESSION ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred_logistic))

print("Precision:",
      precision_score(y_test, y_pred_logistic,
                      average="weighted", zero_division=0))

print("Recall:",
      recall_score(y_test, y_pred_logistic,
                   average="weighted", zero_division=0))

print("F1-Score:",
      f1_score(y_test, y_pred_logistic,
               average="weighted", zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))



print("\n========== DECISION TREE ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred_tree))

print("Precision:",
      precision_score(y_test, y_pred_tree,
                      average="weighted", zero_division=0))

print("Recall:",
      recall_score(y_test, y_pred_tree,
                   average="weighted", zero_division=0))

print("F1-Score:",
      f1_score(y_test, y_pred_tree,
               average="weighted", zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))




print("\n========== NAIVE BAYES ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred_nb))

print("Precision:",
      precision_score(y_test, y_pred_nb,
                      average="weighted", zero_division=0))

print("Recall:",
      recall_score(y_test, y_pred_nb,
                   average="weighted", zero_division=0))

print("F1-Score:",
      f1_score(y_test, y_pred_nb,
               average="weighted", zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))




print("\n========== SUPPORT VECTOR MACHINE ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred_svm))

print("Precision:",
      precision_score(y_test, y_pred_svm,
                      average="weighted", zero_division=0))

print("Recall:",
      recall_score(y_test, y_pred_svm,
                   average="weighted", zero_division=0))

print("F1-Score:",
      f1_score(y_test, y_pred_svm,
               average="weighted", zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))


from sklearn.metrics import ConfusionMatrixDisplay


# Logistic Regression Confusion Matrix

plt.figure(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_logistic,
    cmap="Blues"
)

plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("images/confusion_matrix_logistic.png")
plt.show()




plt.figure(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_tree,
    cmap="Blues"
)

plt.title("Confusion Matrix - Decision Tree")
plt.tight_layout()
plt.savefig("images/confusion_matrix_decision_tree.png")
plt.show()




plt.figure(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_nb,
    cmap="Blues"
)

plt.title("Confusion Matrix - Naive Bayes")
plt.tight_layout()
plt.savefig("images/confusion_matrix_naive_bayes.png")
plt.show()


plt.figure(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_svm,
    cmap="Blues"
)


models = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'SVM']

accuracies = [
    0.7265135699373695,
    0.5803757828810021,
    0.6722338204592901,
    0.697286012526096
]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

for i, accuracy in enumerate(accuracies):
    plt.text(i, accuracy + 0.02, f"{accuracy:.2%}", ha='center')

plt.show()


model_names = [
    "Logistic Regression",
    "Decision Tree",
    "Naive Bayes",
    "SVM"
]

model_accuracies = [
    logistic_accuracy,
    decision_tree_accuracy,
    naive_bayes_accuracy,
    svm_accuracy
]

print("\n========== MODEL COMPARISON ==========")

for name, accuracy in zip(model_names, model_accuracies):
    print(f"{name}: {accuracy:.4f}")

best_model_index = model_accuracies.index(max(model_accuracies))

print("\n========== BEST MODEL ==========")
print("Best Model:", model_names[best_model_index])
print("Best Accuracy:", f"{model_accuracies[best_model_index]:.4f}")


plt.figure(figsize=(10, 6))

plt.bar(model_names, model_accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for i, accuracy in enumerate(model_accuracies):
    plt.text(i, accuracy + 0.02, f"{accuracy:.2%}", ha="center")

plt.show()


import joblib

joblib.dump(logistic_model, "../model/student_performance_model.pkl")

print("\n========== BEST MODEL SAVED ==========")
print("Logistic Regression model saved successfully.")


sample_student = pd.DataFrame([{
    "Age": 17,
    "Gender": 1,
    "Ethnicity": 0,
    "ParentalEducation": 2,
    "StudyTimeWeekly": 10.0,
    "Absences": 5,
    "Tutoring": 1,
    "ParentalSupport": 2,
    "Extracurricular": 1,
    "Sports": 1,
    "Music": 0,
    "Volunteering": 0
}])

prediction = logistic_model.predict(sample_student)

print("Predicted Grade Class:", prediction[0])