import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import joblib
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data_file):
    df = pd.read_csv(data_file)

    label_encoder = LabelEncoder()

    df['name'] = label_encoder.fit_transform(df['name'])

    X = df[["name", "ageCat"]]
    y = df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_filename):
    joblib.dump(model, model_filename)
    print("Model saved successfully.")

if __name__ == "__main__":
    data_file = "plantDiseases.csv"
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data_file)
    trained_model = train_model(X_train, y_train)
    save_model(trained_model, "plantModel.joblib")
    joblib.dump(label_encoder, "label_encoder.joblib")
