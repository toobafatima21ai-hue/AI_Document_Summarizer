from collections import Counter


def word_frequency(words):

    return Counter(words).most_common(15)


def top_keywords(words):

    return Counter(words).most_common(10)