import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import string
from tqdm import tqdm
import time
import pdb
pd.options.mode.chained_assignment = None

class TFIDF:
    #given a cleaned pandas dataframe, creates a pd dataframe with the tfidf of the data
    def __init__(self, data):
        self.data = data

    def tfidf(self):
        text_column = list(self.data.loc[:, 'text'])
        # text_column = list(self.data)
        # print(text_column)
        # standardize the words in each clinical note to lower case without punctuation
        for i in range(len(text_column)):
            text = text_column[i]
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ''.join([i for i in text if not i.isdigit()])
            text = text.lower()
            text_column[i] = text

        # apply tfidf to the data
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(text_column)
        feature_names = vectorizer.get_feature_names_out()
        feature_names = list(feature_names)
        dense = vectors.todense()
        denselist = dense.tolist()
        outdf = pd.DataFrame(denselist, columns=feature_names)

        # prune stopwords which have no deeper meaning
        for stopword in stopwords.words('english'):
            if stopword in outdf.columns:
                outdf.drop(stopword, inplace=True, axis=1)
        
        sum = outdf.sum()
        alpha = 1

        sumList = sum.values.tolist()
        df1 = outdf
        df1.loc[len(df1)] = sumList
        df2 = df1.loc[:, (df1.iloc[259] > alpha)]
        df2.drop(axis=0, index=len(df2) - 1, inplace=True)
        outdf = df2

        return outdf
            