import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, target_column='GradeClass'):
    X = data.drop(columns=['StudentID', target_column])
    y = data[target_column]
    
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    return X, y, numeric_columns

def scale_numeric_columns(X_train, X_test, numeric_columns):
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    return X_train, X_test, scaler
