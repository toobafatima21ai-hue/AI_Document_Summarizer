import nltk
import re

from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download("stopwords")


def preprocess_text(text):

    # Lowercase text
    text = text.lower()

    # Split into sentences without NLTK
    sentences = [
        s.strip()
        for s in re.split(r'[.!?]+', text)
        if s.strip()
    ]

    # Split into words without NLTK
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    return filtered_words, sentences
