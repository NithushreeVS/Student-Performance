import streamlit as st
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, scale_numeric_columns
from src.model_training import get_model, train_model, evaluate_model
from src.prediction import create_new_data, make_prediction
from sklearn.model_selection import train_test_split

# Load the dataset
st.title("Student Prediction")

data = load_data('data/Student_performance_data.csv')
st.subheader("Student Performance Data")
st.write(data.head())

# Preprocess data
X, y, numeric_columns = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, scaler = scale_numeric_columns(X_train, X_test, numeric_columns)

# Model selection
model_choice = st.selectbox("Choose a model", ['Logistic Regression', 'Random Forest', 'K-Nearest Neighbors'])
n_neighbors = 5
if model_choice == 'K-Nearest Neighbors':
    n_neighbors = st.slider('Select number of neighbors for KNN', 1, 15, 5)

model = get_model(model_choice, n_neighbors)
model = train_model(model, X_train, y_train)

# Evaluate the model
accuracy, report = evaluate_model(model, X_test, y_test)
st.write(f"Accuracy with {model_choice}: {accuracy:.2f}")
st.text("Classification Report:")
st.text(report)

# New prediction
st.subheader("Make a New Prediction")
Age = st.number_input("Age", min_value=15, max_value=18, step=1)
Gender = st.selectbox("Gender (1: Male, 0: Female)", [1, 0])
Ethnicity = st.selectbox("Ethnicity (0: Group 1, 1: Group 2, 2: Group 3, etc.)", [0, 1, 2, 3, 4])
ParentalEducation = st.selectbox("Parental Education (0: No formal, 1: High school, etc.)", [0, 1, 2, 3, 4])
StudyTimeWeekly = st.slider("Study Time Weekly", 0.0, 20.0, 10.0)
Absences = st.number_input("Absences", min_value=0, max_value=30, step=1)
Tutoring = st.selectbox("Tutoring (1: Yes, 0: No)", [1, 0])
ParentalSupport = st.selectbox("Parental Support", [1, 2, 3, 4])
Extracurricular = st.selectbox("Extracurricular (1: Yes, 0: No)", [1, 0])
Sports = st.selectbox("Sports (1: Yes, 0: No)", [1, 0])
Music = st.selectbox("Music (1: Yes, 0: No)", [1, 0])
Volunteering = st.selectbox("Volunteering (1: Yes, 0: No)", [1, 0])
GPA = st.number_input("GPA", min_value=0.0, max_value=4.0, step=0.01)

# Make new prediction
new_data = create_new_data(Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, Tutoring, ParentalSupport, Extracurricular, Sports, Music, Volunteering, GPA)
if st.button("Predict Grade Class"):
    prediction = make_prediction(model, new_data, scaler, numeric_columns)
    st.write(f"The predicted Grade Class using {model_choice} is: {prediction[0]}")

