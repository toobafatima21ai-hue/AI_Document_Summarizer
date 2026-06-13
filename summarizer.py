from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
import re


def frequency_summary(text, percentage=30):

    # Split text into sentences without NLTK
    sentences = [
        s.strip()
        for s in re.split(r'[.!?]+', text)
        if s.strip()
    ]

    words = word_tokenize(text.lower())

    freq = Counter()

    for word in words:
        if word.isalnum():
            freq[word] += 1

    sentence_scores = {}

    for sentence in sentences:

        for word in word_tokenize(sentence.lower()):

            if word in freq:

                sentence_scores[sentence] = (
                    sentence_scores.get(sentence, 0)
                    + freq[word]
                )

    count = max(
        1,
        int(len(sentences) * percentage / 100)
    )

    summary_sentences = heapq.nlargest(
        count,
        sentence_scores,
        key=sentence_scores.get
    )

    return " ".join(summary_sentences), sentence_scores


def tfidf_summary(text, percentage=30):

    # Split text into sentences without NLTK
    sentences = [
        s.strip()
        for s in re.split(r'[.!?]+', text)
        if s.strip()
    ]

    if len(sentences) == 0:
        return "", {}

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform(sentences)

    scores = tfidf.sum(axis=1)

    sentence_scores = {}

    for i, sentence in enumerate(sentences):
        sentence_scores[sentence] = float(scores[i, 0])

    count = max(
        1,
        int(len(sentences) * percentage / 100)
    )

    summary_sentences = heapq.nlargest(
        count,
        sentence_scores,
        key=sentence_scores.get
    )

    return " ".join(summary_sentences), sentence_scores
