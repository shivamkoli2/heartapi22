from flask import Flask, request, jsonify
from joblib import load
import numpy as np

# Create Flask application
app = Flask(__name__)

loaded_model = load("heart.joblib")

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()

     # Extract numerical values from the dictionary
    features = [data[key] for key in data]
    
    # Convert features to numpy array
    X = np.array([features])

    predictions = loaded_model.predict(X)

    return jsonify({'predictions':predictions.tolist()})

if __name__ == '__main__':
    app.run(debug = True)
