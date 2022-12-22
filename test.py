import numpy as np 
from Preprocess import Preprocess

preprocessing = Preprocess()

tfidf_vectorizer=preprocessing.readFile("../dumps/tfidf.pickle")
clf = preprocessing.readFile("../dumps/clf.pickle")

sample=[input("Enter a News: ")]
sample=tfidf_vectorizer.transform(sample).toarray()

print(sample)
pred=clf.predict(sample)

print(pred)

label="Fake"
if(pred[0]==1):
    label="Real"    
print(label)