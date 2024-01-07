# Flask App
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import logging

model = pickle.load(open('model.sav', 'rb'))
feature_names = None  # Initialize feature names as None

# Check if the model has feature names attribute
if hasattr(model, 'feature_names'):
    feature_names = model.feature_names

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.json.get('features')  # Update to use JSON data

        if not features:
            return jsonify({'error': 'No features found in the data'})

        # Convert the features to a DataFrame
        features_df = pd.DataFrame([features])

        if feature_names:
            # If model has feature names, map the incoming features to them
            features_mapped = {feature_names[key]: value for key, value in features_df.iloc[0].items()}
        else:
            # If no feature names, use the features as they are
            features_mapped = features_df.iloc[0].to_dict()

        # Convert the mapped features to a DataFrame
        features_mapped_df = pd.DataFrame([features_mapped])

        # Reorder columns to match the order during model training
        if feature_names:
            features_mapped_df = features_mapped_df[feature_names]

        print(features_mapped_df)
        prediction = model.predict(features_mapped_df)
        print(prediction.tolist())
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logging.exception('An error occurred during prediction:')
        return jsonify({'error': 'An error occurred during prediction.', 'details': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
