import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
st.title("Student Prediction")

# Load the CSV file directly from the path
data = pd.read_csv('data/Student_performance_data.csv')

# Display the dataset
st.subheader("Student Performance Data")
st.write(data.head())

# Feature selection and preprocessing
target_column = 'GradeClass'

# Drop the StudentID column since it is not relevant for prediction
X = data.drop(columns=['StudentID', target_column])
y = data[target_column]

# Check for numeric columns
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
st.write(f"Numeric columns: {numeric_columns}")

# Scaling the numeric columns
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_choice = st.selectbox("Choose a model", ['Logistic Regression', 'Random Forest', 'K-Nearest Neighbors'])

if model_choice == 'Logistic Regression':
    model = LogisticRegression(max_iter=1000)
elif model_choice == 'Random Forest':
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    n_neighbors = st.slider('Select number of neighbors for KNN', 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

# Train the selected model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy with {model_choice}: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Input data for new prediction
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

# Create a new data point for prediction
new_data = pd.DataFrame({
    'Age': [Age],
    'Gender': [Gender],
    'Ethnicity': [Ethnicity],
    'ParentalEducation': [ParentalEducation],
    'StudyTimeWeekly': [StudyTimeWeekly],
    'Absences': [Absences],
    'Tutoring': [Tutoring],
    'ParentalSupport': [ParentalSupport],
    'Extracurricular': [Extracurricular],
    'Sports': [Sports],
    'Music': [Music],
    'Volunteering': [Volunteering],
    'GPA': [GPA]
})

# Scale the new data
new_data[numeric_columns] = scaler.transform(new_data[numeric_columns])

if st.button("Predict Grade Class"):
    prediction = model.predict(new_data)
    st.write(f"The predicted Grade Class using {model_choice} is: {prediction[0]}")
