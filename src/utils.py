def log_message(message):
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def calculate_metrics(y_true, y_pred):
    """Calculates and returns accuracy, precision, recall, and F1 score."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def save_model(model, filepath):
    """Saves the trained model to the specified filepath."""
    import joblib
    joblib.dump(model, filepath)

def load_model(filepath):
    """Loads a model from the specified filepath."""
    import joblib
    return joblib.load(filepath)