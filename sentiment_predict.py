from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
import string
import nltk.stem as stem
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

documents = []

for category in movie_reviews.categories():
    for file_id in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(file_id), category))


def remove_stop_words_and_stem(review):
    # initialize stopwords list
    stop_words = set(stopwords.words("english"))
    # initialize stemmer
    wordnet_lemmatizer = stem.WordNetLemmatizer()
    # remove punctuation
    no_punctuation = [word.lower() for word in review
                      if word not in string.punctuation]
    # remove stop words
    no_stop_words = [word for word in no_punctuation if word not in stop_words]
    # stem words
    stemmed = [wordnet_lemmatizer.lemmatize(word) for word in no_stop_words]
    # remove words less than 3 characters long
    final = [word for word in stemmed if len(word) > 2]
    return " ".join(f for f in final)


texts = []
labels = []
for pair in documents:
    document = list(pair[0])
    new = remove_stop_words_and_stem(document)
    texts.append(new)
    labels.append(pair[1])
labels = np.array(labels)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(texts)
Y_train = labels

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)


def predict_sentiment(text):
    tokens = word_tokenize(text)
    processed = remove_stop_words_and_stem(tokens)
    vectorized_data = vectorizer.transform([processed])
    return logreg.predict(vectorized_data)[0]
