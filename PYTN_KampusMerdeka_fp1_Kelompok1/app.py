#importing necessary libraries
from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import joblib
import pickle

df = pd.read_csv('dataset/rideshare_kaggle.csv')
unique_cab_types = df['cab_type'].unique()
unique_name = df['name'].unique()

cab_type_options = "\n".join([f'<option value="{cab_type}">{cab_type}</option>' for cab_type in unique_cab_types])
name_options = "\n".join([f'<option value="{unique_name}">{unique_name}</option>' for unique_name in unique_name])

html_form = f"""
<form id="prediction-form">
    <label for="distance">Distance (in km):</label>
        <input type="number" name="distance" placeholder="Distance" step="0.01" required>
        <br>

    <label for="surge_multiplier">Surge Multiplier:</label>
    <input type="text" name="surge_multiplier" placeholder="Surge Multiplier" required>
    <br>

    <label for="cab_type">Cab Type:</label>
    <select name="cab_type" required>
        {cab_type_options}
    </select>
    <br>

    <label for="name">name:</label>
    <select name="name" required>
        {name_options}
    </select>
    <br>

    <!-- Add other input fields as needed -->

    <button type="button" id="predict-button" onclick="predict()">Predict</button>
</form>
"""


app = Flask(__name__)
with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html', form = html_form)

@app.route('/model', methods=['POST'])
def predict():
    try:
        # Extract features from the JSON request
        data = request.get_json()
        distance = float(data['distance'])
        surge_multiplier = float(data['surge_multiplier'])
        cab_type = float(data['cab_type'])
        name = float(data['name'])


        # Make a prediction using trained model
        prediction = model.predict([[distance, surge_multiplier, cab_type, name]])

        # Extract the prediction result (assuming a single prediction)
        output = round(prediction[0], 2)

        return jsonify({'prediction': output})

    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == "__main__":
    app.run()
