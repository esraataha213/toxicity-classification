import re
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

app = Flask(__name__)

model = xgb.XGBRegressor()
model.load_model('xgb.json')

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text_string):
    text_string = text_string.lower()  
    text_string = re.sub('[^A-Za-z0-9]+', ' ', text_string)  
    x = text_string.split()
    new_text = []
    for word in x:
        if word not in stop_words:
            new_text.append(stemmer.stem(word)) 
    text_string = ' '.join(new_text)
    return text_string

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        preprocessed_text = preprocess(text)

        vectorized_text = vectorizer.transform([preprocessed_text])

        prediction = model.predict(vectorized_text)

        toxicity_class = 'toxic' if prediction[0] >= 0.5 else 'non-toxic'

        return jsonify({'toxicity_level': float(prediction[0]), 'toxicity_class': toxicity_class})


if __name__ == '__main__':
    app.run(debug=True)
