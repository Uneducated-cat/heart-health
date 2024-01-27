from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import logging

model = pickle.load(open('/Users/dheeraj/Desktop/coding/project/heart-health/model.sav', 'rb'))
feature_names = None

if hasattr(model, 'feature_names'):
    feature_names = model.feature_names

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.json.get('features')  
        if not features:
            return jsonify({'error': 'No features found in the data'})

        
        features_df = pd.DataFrame([features])

        if feature_names:
            features_mapped = {feature_names[key]: value for key, value in features_df.iloc[0].items()}
        else:
            features_mapped = features_df.iloc[0].to_dict()

        features_mapped_df = pd.DataFrame([features_mapped])

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
