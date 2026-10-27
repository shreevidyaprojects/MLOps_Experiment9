
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

FEATURES = ["StudyHours", "Attendance", "PreviousMarks"]
TARGET = "FinalMarks"


def load_data(file_path):
    data = pd.read_csv(file_path)

    required = FEATURES + [TARGET]

    if not set(required).issubset(data.columns):
        raise ValueError("Dataset is missing required columns.")

    if data.empty:
        raise ValueError("Dataset is empty.")

    return data


def train_model(data):
    required = FEATURES + [TARGET]

    if not set(required).issubset(data.columns):
        raise ValueError("Dataset is missing required columns.")

    if data.empty:
        raise ValueError("Cannot train on an empty dataset.")

    if data[required].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    model.fit(data[FEATURES], data[TARGET])
    return model


def predict_marks(model, study_hours, attendance, previous_marks):
    student = pd.DataFrame([{
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PreviousMarks": previous_marks
    }])

    return model.predict(student)
