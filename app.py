from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# ---------------- LOAD MODEL FILES ---------------- #
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
FEATURES = pickle.load(open("features.pkl", "rb"))  

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    message = None
    alert_class = None

    if request.method == "POST":
        try:
        
            input_data = {}
            for feature in FEATURES:
                value = request.form.get(feature)
                if value is None or value == "":
                    input_data[feature] = 0.0
                else:
                    input_data[feature] = float(value)

            
            input_df = pd.DataFrame([input_data], columns=FEATURES)

            # Scale
            input_scaled = scaler.transform(input_df)

            # Predict
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled).max()

            confidence = round(prob * 100, 2)

            if pred == 1:
                prediction = "Malignant (Higher Risk)"
                alert_class = "alert-danger"
            else:
                prediction = "Benign (Lower Risk)"
                alert_class = "alert-success"

        except Exception as e:
            prediction = "Error"
            message = "Invalid input or processing error. Please check values and try again."
            alert_class = "alert-warning"
            print("ERROR:", e)

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        alert_class=alert_class
    )


@app.route("/awareness")
def awareness():
    return render_template("awareness.html")


@app.route("/measurements")
def measurements():
    return render_template("measurements.html")


@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")


@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)