import joblib
import spacy
import numpy as np
import re
import emoji

# Load saved bundle
bundle = joblib.load("models/tweet_emotion_model.joblib")
clf = bundle["clf"]
nlp = spacy.load(bundle["spacy_model"], exclude=["tagger","parser","ner","lemmatizer"])

# Cleaning function (same as training)
def clean_tweet(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = emoji.demojize(t, delimiters=(" ", " "))
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Prediction function
def predict_emotion(texts):
    if isinstance(texts, str):
        texts = [texts]
    texts_clean = [clean_tweet(t) for t in texts]
    X_vec = np.vstack([doc.vector for doc in nlp.pipe(texts_clean, batch_size=256)])
    return clf.predict(X_vec)

# Example usage
examples = [
    "I am so happy today!! 😍",
    "This is frustrating and makes me angry...",
    "Feeling low and sad, nothing works out."
]

print(predict_emotion(examples))
