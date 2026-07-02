import sqlite3
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load dataset
data = pd.read_csv("phishing_email.csv")
data = data.sample(n=5000, random_state=42)
data['label'] = data['label'].map({1 : 'phishing', 0:'safe'})

print(data)

emails = data["text_combined"]
labels = data["label"]

X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2)

#convert text to numbers
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#train model
model = RandomForestClassifier()
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)
print(f"\nModel Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

#user input
new_email = input("\nEnter an email to check: ")

#predict
new_X = vectorizer.transform([new_email])
prediction = model.predict(new_X)[0] 

print("\nPrediction:", prediction)

#save result
conn = sqlite3.connect("emails.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS results (email TEXT, prediction TEXT)")
cursor.execute("INSERT INTO results VALUES (?, ?)", (new_email, prediction)) 
conn.commit() 
conn.close()

print("Result saved to database!")

