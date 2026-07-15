print("1")
import pandas as pd
print("2")

from sklearn.model_selection import train_test_split
print("3")

from sklearn.ensemble import RandomForestClassifier
print("4")

from sklearn.metrics import accuracy_score
print("5")

import joblib
print("6")

df = pd.read_csv("Crop_recommendation.csv")
print("7")

X = df.drop("label", axis=1)
print("8")

y = df["label"]
print("9")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("10")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
print("11")

model.fit(X_train, y_train)
print("12")

pred = model.predict(X_test)
print("13")

print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "crop_model.pkl")
print("14 - Done")