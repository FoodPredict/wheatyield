
import flask
import joblib
import numpy as np
from flask import render_template, request

# Load the trained model
model = joblib.load('wheat_yield_model.joblib')

app = flask.Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    temperature = float(request.form['temperature'])
    soil_moisture = float(request.form['soil_moisture'])
    n_kg_ha = float(request.form['n_kg_ha'])
    p_kg_ha = float(request.form['p_kg_ha'])
    k_kg_ha = float(request.form['k_kg_ha'])

    # Create a numpy array from the input values
    features = np.array([[temperature, soil_moisture, n_kg_ha, p_kg_ha, k_kg_ha]])

    # Make a prediction using the loaded model
    prediction = model.predict(features)

    # Format the prediction
    predicted_yield = round(prediction[0], 2)

    # Return the prediction result in the HTML
    return render_template('predict.html',
                           temperature=temperature,
                           soil_moisture=soil_moisture,
                           n_kg_ha=n_kg_ha,
                           p_kg_ha=p_kg_ha,
                           k_kg_ha=k_kg_ha,
                           predicted_yield=predicted_yield)


if __name__ == '__main__':
    # This is used when running locally. Gunicorn or other WSGI server will be used in production.
    # To run locally in Colab, you might need to install ngrok or use Colab's built-in port forwarding
    # app.run(debug=True)
    pass # We won't run the Flask app directly in Colab
