from flask import Flask, request, jsonify, render_template
import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Initialize the Flask application
app = Flask(_name_)

# Download stopwords
nltk.download('stopwords')

# Load the trained model and tokenizer
model = load_model('spam_detection_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocess text function
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    stop_words = set(stopwords.words('english'))  # Set of stopwords
    return " ".join(word.lower() for word in text.split() if word.lower() not in stop_words)

# Define a route for home
@app.route('/')
def home():
    return render_template('index.html')  # Create an 'index.html' template for the homepage

# Define a route for making predictions
@app.route('/predict')
def predict():
    # Get the text input from the form
    text = request.args.get("message")
    
    # Preprocess and tokenize the text input
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    
    # Predict using the trained model
    prediction = (model.predict(padded_sequence) > 0.5).astype('int32')
    label = 'Spam' if prediction == 1 else 'Not Spam'
    
    # Return the prediction as JSON
    response = {'prediction': label}
    return render_template("output.html", response=response)

# Define a route for checking model metrics
@app.route('/metrics', methods=['GET'])
def metrics():
    # Return model accuracy and other metrics as JSON
    test_accuracy = 0.95  # Placeholder: Update with actual test accuracy value
    f1 = 0.90  # Placeholder: Update with actual F1 score
    precision = 0.92  # Placeholder: Update with actual precision score
    recall = 0.88  # Placeholder: Update with actual recall score
    
    return jsonify({
        'test_accuracy': test_accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    })

# Run the app
if _name_ == '_main_':
    app.run(debug=True)