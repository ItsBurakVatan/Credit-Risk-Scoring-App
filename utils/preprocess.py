# utils/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_train(df, target_column):
    df = df.copy()
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X_numeric = X[numeric_cols]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_cols)

    # Eğitim için: ölçekleyici ve kullanılan sütunları döndür
    return X_scaled, y, scaler, numeric_cols


def preprocess_predict(df, scaler, numeric_cols):
    df = df.copy()
    X_numeric = df[numeric_cols]

    X_scaled = pd.DataFrame(scaler.transform(X_numeric), columns=numeric_cols)

    return X_scaled
