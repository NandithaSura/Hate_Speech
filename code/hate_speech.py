# hate_speech.py

import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from googletrans import Translator
from langdetect import detect

# Download NLTK stopwords
nltk.download('stopwords')

class HateSpeechDetection:
    def __init__(self, data_path):
        self.data_path = data_path
        self.stemmer = nltk.SnowballStemmer('english')
        self.stop_words_set = set(stopwords.words('english'))
        self.cv = CountVectorizer()
        self.model = DecisionTreeClassifier()
        self.data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.translator = Translator()

    def translate_to_english(self, text):
        try:
            # Detect the language of the input text
            source_lang = detect(text)
            
            # If text is already in English, return as is
            if source_lang == 'en':
                return text
            
            # Translate to English
            translation = self.translator.translate(text, dest='en')
            return translation.text
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails

    def load_data(self):
        dataset = pd.read_csv(self.data_path)
        dataset['labels'] = dataset['class'].map({0: 'Hate Speech', 1: 'Offensive Language', 2: 'No Hate or Offensive Language'})
        self.data = dataset[['tweet', 'labels']]
    
    def clean_data(self, text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\W', '', text)
        text = ' '.join([word for word in text.split() if word not in self.stop_words_set])
        text = ' '.join([self.stemmer.stem(word) for word in text.split()])
        return text

    def preprocess_data(self):
        self.data['tweet'] = self.data['tweet'].apply(self.clean_data)
        X = np.array(self.data['tweet'])
        Y = np.array(self.data['labels'])
        x = self.cv.fit_transform(X)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, Y, test_size=0.33, random_state=42)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='.1f', cmap='YlGnBu')
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy}')

    def predict(self, text, source_language=None):
        # Only translate if source_language is provided
        if source_language and source_language != 'en':
            try:
                translated_text = self.translator.translate(text, src=source_language, dest='en').text
            except Exception as e:
                print(f"Translation error: {str(e)}")
                translated_text = text  # Fallback to original text if translation fails
        else:
            translated_text = text  # No translation needed
        
        # Clean and process the translated text
        processed_text = self.clean_data(translated_text)
        text_vectorized = self.cv.transform([processed_text]).toarray()
        prediction = self.model.predict(text_vectorized)
        
        return prediction, translated_text
    
    def process_transcription(self, transcription):
        """Process transcribed text and analyze for offensive content"""
        # Clean the transcribed text
        cleaned_text = self.clean_data(transcription)
        
        # Analyze the full text
        vectorized = self.cv.transform([cleaned_text]).toarray()
        overall_prediction = self.model.predict(vectorized)[0]
        
        # Analyze by segments
        sentences = nltk.sent_tokenize(transcription)
        segments = []
        
        for sentence in sentences:
            cleaned_sentence = self.clean_data(sentence)
            vectorized = self.cv.transform([cleaned_sentence]).toarray()
            prediction = self.model.predict(vectorized)[0]
            
            segments.append({
                'text': sentence,
                'prediction': prediction,
                'is_offensive': prediction in ['Hate Speech', 'Offensive Language']
            })
        
        return {
            'overall_prediction': overall_prediction,
            'segments': segments
        }

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()