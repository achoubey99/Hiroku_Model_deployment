import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model using pickle
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input JSON data from the request
        data = request.get_json()

        # Extract the features from the request
        input_data = [
            data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
            data['fbs'], data['restecg'], data['thalach'], data['exang'],
            data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]
        
        # Convert input data to a pandas DataFrame
        input_df = pd.DataFrame([input_data], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])

        # Make prediction
        prediction = model.predict(input_df)
        
        # Return prediction as a JSON response
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
