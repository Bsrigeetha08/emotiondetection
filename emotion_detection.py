import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# 🔹 Path to dataset
DATA_PATH = "twitter_emotions.csv"

# 🔹 Load dataset
df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["id", "text", "label"])
print("🔎 Previewing first 5 rows:\n", df.head())

# 🔹 Check class distribution
print("\n📊 Label distribution before balancing:")
print(df['label'].value_counts())

# 🔹 Balance dataset (upsample minority classes)
balanced_dfs = []
max_count = df['label'].value_counts().max()

for label in df['label'].unique():
    df_label = df[df['label'] == label]
    df_label_upsampled = resample(
        df_label,
        replace=True,                 # sample with replacement
        n_samples=max_count,          # match majority count
        random_state=42
    )
    balanced_dfs.append(df_label_upsampled)

df_balanced = pd.concat(balanced_dfs)

print("\n📊 Label distribution after balancing:")
print(df_balanced['label'].value_counts())

# 🔹 Preprocessing function
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)   # remove links
    text = re.sub(r"@\w+|#\w+", "", text)        # remove @mentions and #hashtags
    text = re.sub(r"[^a-z\s]", "", text)         # keep only letters
    text = re.sub(r"\s+", " ", text).strip()     # remove extra spaces
    return text

# 🔹 Apply preprocessing
df_balanced["clean_text"] = df_balanced["text"].apply(clean_text)

# 🔹 Split data
X = df_balanced["clean_text"]
y = df_balanced["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🔹 Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 🔹 Evaluate
y_pred = model.predict(X_test_tfidf)
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 🔹 Interactive prediction
while True:
    user_input = input("\n✍️ Enter a tweet (or 'quit' to exit): ")
    if user_input.lower() == "quit":
        break
    clean_input = clean_text(user_input)
    user_input_tfidf = vectorizer.transform([clean_input])
    prediction = model.predict(user_input_tfidf)[0]
    print("👉 Detected Emotion:", prediction)
