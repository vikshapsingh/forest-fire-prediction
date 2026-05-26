import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home page
@app.route('/')
def index():
    return render_template('home.html')

# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":

        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]

        scaled_data = standard_scaler.transform(new_data)

        result = ridge_model.predict(scaled_data)

        return render_template(
            'home.html',
            prediction_text=f"Predicted Fire Weather Index: {result[0]}"
        )

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)