import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

def get_x(df, is_train=False):
    df = pd.get_dummies(df, columns=["Pclass", "Sex", "Embarked"], drop_first=True)
    if is_train:
        df = df.dropna(subset=["Age"])
    else:
        df.Age = df.Age.fillna(0)
    x = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare", "SibSp", "Parch"], axis=1)
    if "Survived" in x:
        x = x.drop(["Survived"], axis=1)
    with open("data/scaler/age_scaler", "rb") as fp:
        age_scaler = pickle.load(fp)
        x.Age = age_scaler.fit_transform(x.Age.values.reshape(-1,1))
    return x

def get_x_and_y(df):
    x = get_x(df)
    y = df["Survived"].values
    return x, y

def get_model():
    with open("data/models/rfc_5", "rb") as fp:
        return pickle.load(fp)

def predict(df):
    model = get_model()
    x = get_x(df)
    return model.predict(x)
