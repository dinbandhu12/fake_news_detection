import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from Preprocess import Preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

preprocessing = Preprocess()

df= pd.read_csv("../data/dataset.csv")

X= df["text"]
Y= df["class"]

X_preprocessed = X.copy()

max_len =0
for id,text in enumerate(X):
    text = preprocessing.word_drop(text)
    X_preprocessed[id] =text
    val= text.split(" ")
    if(val>max_len):
        max_len = len(text)

print(max_len)
exit()
tfidf_vectorizer = preprocessing.getTFIDF()
tfidf = tfidf_vectorizer.fit_transform(X).toarray()

preprocessing.writeFile("../dumps/tfidf.pickle",tfidf_vectorizer)

print(tfidf.shape)

X_train, X_test, y_train, y_test = train_test_split(tfidf,Y,test_size=0.25,random_state=0)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

preprocessing.writeFile("../dumps/clf.pickle",clf)


    
y_pred=clf.predict(X_test)
print("Confusion Matrix")
cm=confusion_matrix(y_test, y_pred)

print(cm)
print("Accuracy Score")
print(accuracy_score(y_test, y_pred))

print("Classification Report")
print(classification_report(y_test, y_pred))
