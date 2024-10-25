import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Read the dataset
df = pd.read_csv('amazon_alexa.csv')

# Step 2: Remove null values
df.dropna(inplace=True)

# Step 3: Preprocess the Amazon Alexa reviews
table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Tokenize words
    tokens = word_tokenize(text)

    # Convert words to lower case
    tokens = [word.lower() for word in tokens]

    # Remove punctuation and stop words
    words = [word.translate(table) for word in tokens if word not in stop_words]

    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join words into a single string
    return ' '.join(words)

df['cleaned_reviews'] = df['verified_reviews'].apply(preprocess)

# Step 4: Transform the words into vectors using CountVectorizer
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(df['cleaned_reviews'])

# Step 5: Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, df['feedback'], test_size=0.2, random_state=0)

# Step 6: Train the model 
# a) Multinomial Naïve Bayes Classification
# b) Logistic Regression
# c) KNN Classification

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

model1 = MultinomialNB()
model2 = LogisticRegression()
model3 = KNeighborsClassifier()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Step 7: Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)

print('Multinomial Naïve Bayes Classification')
print('Accuracy: ', accuracy_score(y_test, y_pred1))

print('Logistic Regression')
print('Accuracy: ', accuracy_score(y_test, y_pred2))

print('KNN Classification')
print('Accuracy: ', accuracy_score(y_test, y_pred3))

# Predict the feedback for test data
test_data = ['The sound quality is not good', 'The sound quality is good', 'The sound quality is excellent']
test_data = vectorizer.transform(test_data)
print(model1.predict(test_data))

# Compute Confusion matrix and classification report for each of these models
print('Multinomial Naïve Bayes Classification')
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred1))
print('Classification Report: ', classification_report(y_test, y_pred1))

print('Logistic Regression')
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred2))
print('Classification Report: ', classification_report(y_test, y_pred2))

print('KNN Classification')
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred3))
print('Classification Report: ', classification_report(y_test, y_pred3))



# 9) Report the model with the best accuracy using matplotlib barplot
import matplotlib.pyplot as plt
import numpy as np

models = ['Multinomial Naïve Bayes Classification', 'Logistic Regression', 'KNN Classification']
accuracy = [accuracy_score(y_test, y_pred1), accuracy_score(y_test, y_pred2), accuracy_score(y_test, y_pred3)]

plt.figure(figsize=(10, 5))
plt.bar(models, accuracy)
plt.title('Accuracy of the models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.bar_label(plt.gca().containers[0], fmt='%.3f')
plt.show()

#Multinomial wins