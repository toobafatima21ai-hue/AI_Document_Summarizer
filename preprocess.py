from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


def preprocess_text(text):

    text = text.lower()

    words = word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    filtered_words = [
        word
        for word in words
        if word.isalnum() and word not in stop_words
    ]

    sentences = sent_tokenize(text)

    return filtered_words, sentences