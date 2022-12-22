import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer 
import pickle 
import nltk 
from nltk.corpus import stopwords 
nltk.download('stopwords')

stop_words = set(stopwords.words('english')) 

from nltk.tokenize import word_tokenize


class Preprocess:
    def __init__(self):

        self.ps = PorterStemmer() 

    def word_drop(self,text):
  
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub("\\W"," ",text) 
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)  

        word_tokens = word_tokenize(text)
        x = [w for w in word_tokens if not w in stop_words] 
        x= [ self.ps.stem(words) for words in x]
        x=' '.join(x)  
        return x

    def getTFIDF(self):
        tfidf_vectorizer = TfidfVectorizer(max_features=32767,preprocessor=self.word_drop)
        return tfidf_vectorizer

    def readFile(self,filePath):
        with open(filePath,"rb") as f:
            vector=pickle.load(f)
        return vector

    def writeFile(self,filePath,fileObject):
        with open(filePath,"wb") as f:
            pickle.dump(fileObject,f)
  