import pandas as pd

def create_new_data(Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, Tutoring, ParentalSupport, Extracurricular, Sports, Music, Volunteering, GPA):
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
    return new_data

def make_prediction(model, new_data, scaler, numeric_columns):
    new_data[numeric_columns] = scaler.transform(new_data[numeric_columns])
    return model.predict(new_data)
