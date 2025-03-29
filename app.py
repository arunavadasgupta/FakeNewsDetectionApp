

from flask import Flask, request, jsonify, send_from_directory
import requests
from bs4 import BeautifulSoup
import torch
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification
)
import kagglehub
import os
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.svm import SVC
from nltk.corpus import stopwords
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from nltk.tokenize import word_tokenize
from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image



app = Flask(__name__)
CORS(app)

## Load models and tokenizers mapping
MODEL_MAPPING = {
    "distilbert": (DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
                   DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")),
    "bert": (BertTokenizer.from_pretrained("bert-base-uncased"),
             BertForSequenceClassification.from_pretrained("bert-base-uncased")),
    "roberta": (RobertaTokenizer.from_pretrained("roberta-base"),
                RobertaForSequenceClassification.from_pretrained("roberta-base")),
    "xlnet": (XLNetTokenizer.from_pretrained("xlnet-base-cased"),
              XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")),
    "albert": (AlbertTokenizer.from_pretrained("albert-base-v2"),
               AlbertForSequenceClassification.from_pretrained("albert-base-v2")),
}



nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

output_folder = os.path.join(os.getcwd(), 'user_accessible_models') 
os.makedirs(output_folder, exist_ok=True)  

assets_folder = os.path.join(os.getcwd(), 'assets')
os.makedirs(assets_folder, exist_ok=True)

uploaded_files_folder = os.path.join(os.getcwd(), 'uploaded_files')
os.makedirs(uploaded_files_folder, exist_ok=True)

# Load pre-trained image classification model
image_model_path = os.path.join(output_folder, 'cifake_image_classifier_model.h5')  # Update this to the actual model path
image_model = load_model(image_model_path)

# Function to preprocess text and remove stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in STOPWORDS]
    return ' '.join(tokens)



def fetch_content(url):
    try:
        response = requests.get(url, timeout=10)  # timeout to avoid hanging
        response.raise_for_status()  #  error for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from all <p> tags
        paragraphs = soup.find_all('p')
        content_text = ' '.join(p.get_text() for p in paragraphs)
        print("Fetched content:", content_text[:100])  # first 100 characters for debugging

        return content_text.strip()
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None
    


## Function to classify text using DistilBERT 
def classify_text(text, model, tokenizer):
    print(model)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()
    probability = torch.softmax(logits, dim=1)
    return predicted_class, probability[0][predicted_class].item()


##Function to classify the output based on the text classification
def classify_output(predicted_class, probability):
    """
    Determine classification based on model output.

    :param predicted_class: The predicted class from the model
    :param probability: The probability of the predicted class
    :return: Classification result as "Real", "Fake", or "Uncertain"
    """
    # log the details for debugging
    print(f"Predicted class: {predicted_class}, Probability: {probability}")

    if probability < 0.5:
        return "Uncertain"
    elif predicted_class == 0:
        return "Fake"
    else:
        return "Real"

# Function to save evaluation metrics graphics
def save_evaluation_metrics(y_test, y_pred, X_test_vectorized, model):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Fake', 'Real'])
    plt.yticks(tick_marks, ['Fake', 'Real'])

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    cm_path = os.path.join(assets_folder, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_vectorized)[:, 1], pos_label='REAL')
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    roc_curve_path = os.path.join(assets_folder, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close()


############## Server Routes ####################

@app.route('/classify', methods=['POST'])
def classify():
    model_file = request.files.get('model')
    vectorizer_file = request.files.get('vectorizer')
    csv_file = request.files.get('csv')
    url = request.form.get('url')
    text = request.form.get('text')
    model_type = request.form.get('model_type', 'distilbert').lower()  # Default to distilbert if not provided

    print(f"Incoming model_type: {model_type}")

    # Validate user input
    if not text and not url and not csv_file:
        return jsonify({'error': 'No valid input provided. Please provide a URL, CSV file, or text for classification.'}), 400

    # Handle NLP model classification
    if model_type == 'nlp':
        model_path = os.path.join(output_folder, 'trained_model.pkl')
        vectorizer_path = os.path.join(output_folder, 'tfidf_vectorizer.pkl')

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)

            text_processed = preprocess_text(text)
            tokens = vectorizer.transform([text_processed])
            prediction = model.predict(tokens)
            prediction_proba = model.predict_proba(tokens)[0].max()

            classification_label = "Fake" if prediction[0] == 0 else "Real"

            return jsonify({
                'classification': classification_label,
                'probability': prediction_proba
            })
        else:
            return jsonify({'error': 'Model or vectorizer not found'}), 400

    # Handle Transformer model classification
    elif model_type in MODEL_MAPPING:
        tokenizer, model = MODEL_MAPPING[model_type]

        # Classify text from a URL
        if url:
            print(f"Fetching content from URL: {url}")
            content = fetch_content(url)
            if not content:
                return jsonify({'error': 'Could not fetch content from the URL'}), 400
            text = content
            print("Fetched content:", content[:100])  # Print the first 100 characters for debugging

        # Classify using the Transformer model
        predicted_class, probability = classify_text(text, model, tokenizer)
        classification_result = classify_output(predicted_class, probability)

        return jsonify({
            'classification': classification_result,
            'probability': probability
        })

    # Invalid model type
    else:
        return jsonify({'error': 'Invalid model type selected.'}), 400

    return jsonify({'error': 'Unexpected error. Please check your input and try again.'}), 500




# route to serve image files

@app.route('/assets/<filename>')
def serve_file(filename):
    return send_from_directory(assets_folder, filename)


#route to train the model when user presses train

@app.route('/train', methods=['POST'])
def train_model():
    # Check if a file was uploaded
    if 'file' not in request.files:
        # If no file, download the dataset
        path = kagglehub.dataset_download("rajatkumar30/fake-news")
        print("Path to dataset files:", path)
        
        # Load the dataset
        df = pd.read_csv(os.path.join(path, 'news.csv'))
    else:
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        df = pd.read_csv(file)

    df = df.dropna(subset=['text'], how='any')
    df = df[df['text'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)

    df['cleaned_text'] = df['text'].apply(preprocess_text)

    X = df['cleaned_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model_type = request.form.get('model_type', 'Logistic Regression') 

    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=3000)
    elif model_type == "Random Forest":
        model = RandomForestClassifier()
    elif model_type == "SVM":
        model = SVC(probability=True)
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='REAL')
    recall = recall_score(y_test, y_pred, pos_label='REAL')
    f1 = f1_score(y_test, y_pred, pos_label='REAL')

    save_evaluation_metrics(y_test, y_pred, X_test_vectorized, model)
    
    joblib.dump(model, os.path.join(output_folder, 'trained_model.pkl'))
    joblib.dump(vectorizer, os.path.join(output_folder, 'tfidf_vectorizer.pkl'))

    return jsonify({
        'message': 'Model trained successfully.',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cm': confusion_matrix(y_test, y_pred).tolist(),
        'confusion_matrix_image': '/assets/confusion_matrix.png',
        'roc_curve_image': '/assets/roc_curve.png',
        'model_file': '/assets/trained_model.pkl',  
        'vectorizer_file': '/assets/tfidf_vectorizer.pkl' 
    })

##Route for Image Classification

@app.route('/classify_image', methods=['POST'])
def classify_image():
    file = request.files.get('image')

    # Check if a file was uploaded
    if not file:
        return jsonify({'error': 'No image file provided'}), 400

    # Load and preprocess the image
    img_path = os.path.join(uploaded_files_folder, file.filename)
    file.save(img_path)  

    # Preprocess the image for the model
    img = keras_image.load_img(img_path, target_size=(150, 150))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make a prediction
    predictions = image_model.predict(img_array)
    predicted_class = 'Real' if predictions[0][0] > 0.5 else 'Fake'
    probability_real = predictions[0][0]  # Probability of being 'Real'
    probability_fake = 1 - probability_real  # Probability of being 'Fake'

    return jsonify({'classification': predicted_class,
                    'probability_real': float(probability_real),
                    'probability_fake': float(probability_fake)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)


