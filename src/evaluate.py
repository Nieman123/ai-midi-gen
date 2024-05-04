import logging
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions.round())
    report = classification_report(y_test, predictions.round())
    logging.info(f"Model accuracy: {accuracy}")
    logging.info(f"Classification Report: \n{report}")
    return accuracy, report
