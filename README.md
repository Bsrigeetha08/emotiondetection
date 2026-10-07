# Emotion Detection from Tweets

A machine learning project for detecting emotions in Twitter text using Natural Language Processing (NLP) and machine learning techniques.

## Features

- 🧹 Text preprocessing and cleaning
- ⚖️ Dataset balancing using upsampling
- 🔤 TF-IDF text vectorization
- 🤖 Multinomial Naive Bayes classification
- 📊 Classification report and confusion matrix
- 💬 Interactive emotion prediction for new tweets
- 🧠 Saved-model inference using spaCy
- 😊 Emoji-aware text preprocessing

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- spaCy
- Joblib
- Emoji

## How It Works

The project processes tweet text and uses machine learning techniques to classify the associated emotion.

### Training Pipeline

1. Load the Twitter emotion dataset.
2. Analyze the distribution of emotion labels.
3. Balance the dataset using upsampling.
4. Clean and preprocess the tweet text.
5. Convert the text into numerical features using TF-IDF.
6. Train a Multinomial Naive Bayes classifier.
7. Evaluate the model using a classification report and confusion matrix.
8. Predict emotions from new user-provided tweets.

## Text Preprocessing

The text preprocessing pipeline includes:

- Converting text to lowercase
- Removing URLs
- Removing user mentions
- Handling hashtags
- Removing unnecessary characters
- Removing extra whitespace
- Emoji-aware preprocessing in the inference pipeline

## Project Structure

```text
emotiondetection/
└── emotion-detection-twitter/
    ├── data/
    ├── dataset_final.csv.xlsx
    ├── twitter_emotions.csv
    ├── emotion_detection.py
    ├── inference.py
    └── requirements.txt
