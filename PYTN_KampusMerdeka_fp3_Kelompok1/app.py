from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the Random Forest model and its scaler
with open('rf.pkl', 'rb') as file:
    rf_data = pickle.load(file)
    rf_model = rf_data['model']
    rf_scaler = rf_data['scaler']

# Load the Gradient Boosting model and its scaler
with open('gb.pkl', 'rb') as file:
    gb_data = pickle.load(file)
    gb_model = gb_data['model']
    gb_scaler = gb_data['scaler']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        time = float(request.form['time'])
        serum_creatinine = float(request.form['serum_creatinine'])
        ejection_fraction = float(request.form['ejection_fraction'])
        age = float(request.form['age'])
        serum_sodium = float(request.form['serum_sodium'])

        # Scale the input data using the loaded scalers for both models
        input_data = [[time, serum_creatinine, ejection_fraction, age, serum_sodium]]

        rf_input_data = rf_scaler.transform(input_data)
        gb_input_data = gb_scaler.transform(input_data)

        # Make predictions using the loaded models
        rf_prediction = rf_model.predict(rf_input_data)[0]
        gb_prediction = gb_model.predict(gb_input_data)[0]

        # Return the predictions 
        return jsonify({'rf_prediction': rf_prediction, 'gb_prediction': gb_prediction})

    except ValueError as e:
        return jsonify({'error': str(e)})
    
    
if __name__ == '__main__':
    app.run(debug=True)
