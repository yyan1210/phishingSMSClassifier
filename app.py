import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score

# Load the preprocessed data
df = pd.read_csv('preprocessed_sms.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['sms'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# Save the model to a file
with open('nb_classifier-2.pkl', 'wb') as f:
    pickle.dump(nb_classifier, f)
    pickle.dump(vectorizer, f)  # save the vectorizer as well

# Load the saved model and vectorizer
with open('nb_classifier-2.pkl', 'rb') as f:
    nb_classifier = pickle.load(f)
    vectorizer = pickle.load(f)

# # Load the preprocessed test data
# test_data = pd.read_csv('preprocessed_sms.csv')
# X_test = test_data['sms']
# y_test = test_data['label']

# # Vectorize the test data using the loaded vectorizer
# X_test_vectorized = vectorizer.transform(X_test)

# # Use the loaded classifier to predict the class labels of the test data
# y_pred = nb_classifier.predict(X_test_vectorized)

# # Calculate the accuracy of the classifier
# accuracy = accuracy_score(y_test, y_pred)


def main():
    st.title("Phishing SMS Scam Meter")

    text = st.text_input("Enter a new SMS to be classified:")

    # Set the initial value of prediction to None
    prediction = None

    # Check if the input text is empty
    if not text:
        st.warning("Please enter a valid SMS.")
    else:
        # Preprocess the new data using the same steps as before
        stemmer = PorterStemmer()
        preprocessed_text = text.lower()
        preprocessed_text = re.sub(r'[^\w\s]', '', preprocessed_text)
        stop_words = set(stopwords.words('english'))
        preprocessed_text = [word for word in preprocessed_text.split() if word not in stop_words]
        preprocessed_text = [stemmer.stem(word) for word in preprocessed_text]
        preprocessed_text = " ".join(preprocessed_text)

        # Vectorize the preprocessed new data using the fitted vectorizer
        try:
            vectorized_text = vectorizer.transform([preprocessed_text])
        except NotFittedError:
            st.error("Model not fitted. Please run the model training script.")
            return

        # Use the trained classifier to predict the class labels of the new data
        prediction = nb_classifier.predict(vectorized_text)

    # Display the predicted label
    if prediction is not None:
        y_pred = nb_classifier.predict(vectorizer.transform(X_test))
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Prediction: {prediction[0]}")
        st.write(f"Accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    main()
