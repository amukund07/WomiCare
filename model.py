import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
df = df.drop(["id", "Unnamed: 32","smoothness_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "texture_se",
    "perimeter_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "smoothness_worst",
    "compactness_worst",
    "symmetry_worst",
    "fractal_dimension_worst"], axis=1)

# binary encoding
df["diagnosis"] = df["diagnosis"].map({'B': 0, 'M': 1})

# Splitting data set
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standaraization
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train[:] = scaler.transform(X_train)
X_test[:] = scaler.transform(X_test)

# Trainig model
model = RandomForestClassifier(
    n_estimators=300,
    class_weight={0:1, 1:2},
    random_state=42
)

model.fit(X_train, Y_train)


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    for col in X_train.columns:
        if col not in input_df.columns:
            input_df[col] = 0  

    input_df = input_df[X_train.columns]
    input_df = pd.DataFrame(scaler.transform(input_df), columns=X_train.columns)
    pred_class = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df).max()
    pred_label = "Malignant" if pred_class == 1 else "Benign"

    return pred_label, pred_prob


import pickle

# Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature order 
with open("features.pkl", "wb") as f:
    pickle.dump(list(X_train.columns), f)

print(" Model, scaler, and features saved successfully!")








# if __name__ == "__main__":

#     single_input = {
#         "radius_mean": 14.2,
#         "texture_mean": 18.3,
#         "perimeter_mean": 92.5,
#         "area_mean": 600.1,
#         "compactness_mean": 0.1,
#         "concavity_mean": 0.05,
#         "concave points_mean": 0.05,
#         "radius_worst": 16.5,
#         "texture_worst": 25.0,
#         "perimeter_worst": 107.0,
#         "area_worst": 800.0,
#         "concavity_worst": 0.15,
#         "concave points_worst": 0.1,
#     }

#     pred_label, pred_prob = predict_input(single_input)
#     print(f"Prediction: {pred_label}, Confidence: {pred_prob:.2f}")
