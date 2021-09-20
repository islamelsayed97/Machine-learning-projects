import re
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from os import path

class IndustryClassification:
    def __init__(self):
        self.tf_idf      = TfidfVectorizer()
        self.oversample  = SMOTE(sampling_strategy={1:1529, 2:1500, 3:1300, 4:1000}, random_state=0)
        self.clf         = MultinomialNB()
        self.is_trained  = False
        self.result      = {'1':'IT', '2':'Marketing', '3':'Education', '4':'Accountancy'}
        self.stop_words  = stopwords.words('english')
        self.stop_words.remove('it')# removing 'IT'
        
    def __get_data(self): # private method because I will use it in this class only
        self.Job_titles = pd.read_csv('Job titles and industries.csv')
        
    def __clean(self, text): # private method because I will use it in this class only
        # remove all non alphabetic characters
        text = re.sub('[^A-Za-z]', ' ', text)
        
        # make all words in lowercase and split the text in words list
        words = text.lower().split()
        
        # loop for every word in words and check if it isn't in 'stop_words', then add it to 'wanted_words' list
       
        wanted_words = []
        for word in words:
            if word not in self.stop_words:
                # there are job titles have salary in our data, like £55k after removeing the number and the 
                # special character £, there is only 'k', I want to remove this also.
                # so I will remove any word that have only one character
                if len(word) > 1:
                    wanted_words.append(word)
                
        # join the words again to put them in a cleanes text.
        return ' '.join(wanted_words)
    
    def __preprocess(self): # private method because I will use it in this class only
        self.__get_data()
        industry_mapping                   = {'IT':1, 'Marketing':2, 'Education':3, 'Accountancy':4}
        self.Job_titles                    = self.Job_titles.drop_duplicates()
        self.Job_titles['industry']        = self.Job_titles['industry'].map(industry_mapping)
        self.Job_titles['clean job title'] = self.Job_titles['job title'].apply(self.__clean)
        self.X                             = self.tf_idf.fit_transform(self.Job_titles['clean job title']).toarray()
        self.y                             = self.Job_titles.industry
        self.X_smote, self.y_smote         = self.oversample.fit_resample(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_smote, self.y_smote, test_size = 0.2, random_state=0, stratify=self.y_smote)

    def train(self):
        if not path.exists("model.joblib"): # if this file is not exist, this meaning that, this is the first time and we have to train the model
            self.__preprocess()
            self.clf.fit(self.X_train, self.y_train)
            self.is_trained = True
            dump(self.clf, 'model.joblib') # save the trained model to reuse it next times
            dump(self.tf_idf , 'tf_idf.joblib')  # also save tf_idf object to use it in preprocessing to predict new titles
        else:           # if the file is exist, this meaning that we already have a trained model and we can use it directly
            self.__preprocess()
            self.clf = load('model.joblib')  # if the model and to tf_idf were already saved, load them instead of training the model again
            self.tf_idf = load('tf_idf.joblib')

    def evaluate(self):
        if self.is_trained == False: # if the model is not trained, train it first and then make the evaluation, otherwise, make the evaluation directly
            self.train()
        y_pred = self.clf.predict(self.X_test)

        return round(f1_score(self.y_test, y_pred, average='weighted')*100, 2)
            
    def predict(self, text):
        if self.is_trained==False: # if the model is not trained, train it first and then make the prediction, otherwise, make the prediction directly
            self.train()
        test = self.__clean(text)
        test = self.tf_idf.transform([test]).toarray()
        res  = str(self.clf.predict(test)[0])

        return self.result[res]