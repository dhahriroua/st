import joblib


def predict(data):
    mlp = joblib.load("monmodelmlp.sav")
    return mlp.predict(data)
