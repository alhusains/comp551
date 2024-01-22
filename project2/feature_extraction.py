import multiprocessing
import string
from enum import Enum

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureType(Enum):
    ALL_WORDS = 1
    EMOTION_WORDS = 2
    SENTENCE_EMOTION = 3
    PART_OF_SPEECH = 4
    TEXT_LENGTH = 5
    SENTENCE_LENGTH = 6


def get_features(X, feature_type, vectorizer=None):
    if feature_type is FeatureType.ALL_WORDS:
        features, vectorizer = get_features_words(X, feature_type, vectorizer, False)
    elif feature_type is FeatureType.EMOTION_WORDS:
        features, vectorizer = get_features_words(X, feature_type, vectorizer, True)
    elif feature_type is FeatureType.SENTENCE_EMOTION:
        features = get_features_sentence_emotion(X)
    elif feature_type is FeatureType.PART_OF_SPEECH:
        features, vectorizer = get_features_pos(X, vectorizer)
    elif feature_type is FeatureType.TEXT_LENGTH:
        features = get_features_text_length(X)
    elif feature_type is FeatureType.SENTENCE_LENGTH:
        features = get_features_sentence_length(X)
    else:
        features = X

    return features, vectorizer


def get_features_words(X, feature_type=None, vectorizer=None, preprocessing=False, vocabulary_prop=1.0):
    if preprocessing:
        X = preprocess(X, feature_type)

    if 0 < vocabulary_prop < 1:
        vocabulary_size = len({
            token
            for text in X
            for token in word_tokenize(text)
        })
        max_features = int(vocabulary_size * vocabulary_prop)
    else:
        max_features = None

    if vectorizer is None:
        vectorizer = get_vectorizer(max_features)
        features = vectorizer.fit_transform(X)
    else:
        features = vectorizer.transform(X)

    return features, vectorizer


def get_features_sentence_emotion(X, num_sentences=25, analyzer=SentimentIntensityAnalyzer()):
    features = []

    for text in X:
        sentences = sent_tokenize(text)[:num_sentences]
        sentence_emotion = [0] * num_sentences
        sentence_emotion[:len(sentences)] = (
            analyzer.polarity_scores(sentence)['compound']
            for sentence in sentences
        )
        features.append(sentence_emotion)

    return features


def get_features_pos(X, vectorizer=None):
    X_pos = [
        ' '.join(
            tag
            for sentence in sent_tokenize(text)
            for _, tag in pos_tag(word_tokenize(sentence))
        )
        for text in X
    ]

    if vectorizer is None:
        vectorizer = get_vectorizer()
        features = vectorizer.fit_transform(X_pos)
    else:
        features = vectorizer.transform(X_pos)

    return features, vectorizer


def get_features_text_length(X):
    features = [[len(text)] for text in X]
    return features


def get_features_sentence_length(X, num_sentences=25):
    features = []

    for text in X:
        sentences = sent_tokenize(text)[:num_sentences]
        sentence_length = [0] * num_sentences
        sentence_length[:len(sentences)] = (len(sentence) for sentence in sentences)
        features.append(sentence_length)

    return features


def get_vectorizer(max_features=None, use_counts=False):
    if use_counts:
        vectorizer = CountVectorizer(max_features=max_features)
    else:
        vectorizer = TfidfVectorizer(max_features=max_features)
    return vectorizer


def get_punctuation_table():
    punctuation_table = str.maketrans('', '', string.punctuation)
    return punctuation_table


punctuation_table = get_punctuation_table()


def get_stemmer():
    stemmer = PorterStemmer()
    return stemmer


stemmer = get_stemmer()


def get_stop_words():
    stop_words = set(stopwords.words('english'))
    return stop_words


stop_words = get_stop_words()


def get_emotion_words(emotion_threshold=0.1):
    analyzer = SentimentIntensityAnalyzer()
    lex_dict = analyzer.make_lex_dict()
    emotion_words = {
        stemmer.stem(word)
        for word, measure in lex_dict.items()
        if abs(measure) >= emotion_threshold
    }
    return emotion_words


emotion_words = get_emotion_words()


def preprocess(X, feature_type=None, num_workers=4):
    X_pre = []

    with multiprocessing.Pool(num_workers) as pool:
        futures = []

        m = len(X) // num_workers
        for k in range(num_workers):
            i = m * k
            j = m * (k + 1) if k + 1 < num_workers else len(X)
            future = pool.apply_async(preprocess_helper, (X[i:j], feature_type))
            futures.append(future)

        for future in futures:
            X_pre.extend(future.get())

    return X_pre


def preprocess_helper(X, feature_type=None):
    X_pre = []

    for text in X:
        # split into tokens
        tokens = word_tokenize(text)

        # remove punctuation
        tokens = [token.translate(punctuation_table) for token in tokens]

        # remove non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]

        # convert to lowercase
        tokens = [token.lower() for token in tokens]

        # remove stop words
        tokens = [token for token in tokens if token not in stop_words]

        # extract word stems
        tokens = [stemmer.stem(token) for token in tokens]

        # extract emotion words
        if feature_type is FeatureType.EMOTION_WORDS:
            tokens = [token for token in tokens if token in emotion_words]

        X_pre.append(' '.join(tokens))

    return X_pre
