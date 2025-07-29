# model/trainer.py
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config.config import MODEL_PATH

def train_model(X, y, scaler):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    # Scaler'ı da kaydet
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return model, report

