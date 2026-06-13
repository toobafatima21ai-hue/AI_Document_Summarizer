import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# download required NLTK data (safe for Streamlit)
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):

    # lowercase
    text = text.lower()

    # sentence splitting (safe for cloud)
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # word tokenization
    words = word_tokenize(text)

    # stopword removal
    stop_words = set(stopwords.words("english"))

    filtered_words = [
        word for word in words
        if word.isalnum() and word not in stop_words
    ]

    return filtered_words, sentences
