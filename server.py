from flask import Flask, render_template, request
import xgboost as xgb
import joblib
import os
import trafilatura
from sentence_transformers import SentenceTransformer
from bs4 import UnicodeDammit

app = Flask(__name__)

# Load your XGBoost model
xgboost_model = joblib.load('model/xgboost_model.pkl')
catboost_model = joblib.load('model/catboost_model.pkl')

# Initialize Sentence Transformer model
sentence_transformer_model = SentenceTransformer('aditeyabaral/sentencetransformer-xlm-roberta-base')


def html_to_text(html_content):
    text = trafilatura.extract(UnicodeDammit(html_content).unicode_markup)
    return text

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    prediction_result = "Phishing"  # Default value

    if 'htmlFile' not in request.files:
        return "No file part"

    file = request.files['htmlFile']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to a temporary location

    file_path = 'test\\' + file.filename
    file.save(file_path)

    # Extract text content from HTML file
    with open( file_path, 'r', encoding='utf-8') as embeddings_file:
        html_content = embeddings_file.read()
    text_content = html_to_text(html_content)

    # Encode text content using Sentence Transformer
    embedding = sentence_transformer_model.encode([text_content])

    # Make prediction using the XGBoost model
    xgboost_prediction = xgboost_model.predict(embedding)[0]
    catboost_prediction = catboost_model.predict(embedding)[0]

    # Map prediction to a human-readable result

    prediction_result = "Phishing" if \
        catboost_prediction == 1 or xgboost_prediction == 1\
        else "Legitimate"
    return f"{file_path} is {prediction_result}"


if __name__ == '__main__':
    app.run(debug=True, port=5050)
