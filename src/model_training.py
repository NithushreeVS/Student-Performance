from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def get_model(model_choice, n_neighbors=5):
    if model_choice == 'Logistic Regression':
        return LogisticRegression(max_iter=1000)
    elif model_choice == 'Random Forest':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return KNeighborsClassifier(n_neighbors=n_neighbors)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report
