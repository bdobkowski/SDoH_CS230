import numpy as np
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from matplotlib import pyplot as plt
import torch
import re
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

# from tfidf import TFIDF
nltk.download('punkt')
nltk.download('stopwords')

eng_stopwords = stopwords.words('english')

# weak_label_words = ['hunger','hungry','nutrition','food','skinny','eat','eating']
weak_label_words = ['nutrition','food','weight']

# these are the top n words after the stopwords in the positive labeled data
# weak_label_words = ['mg', 'patient', 'pain', 'daily', 'history', 'pt', 'md', 'labs', 'per', 'tablet']

def load_data(data_path, unstructured_data=None):
    
	df, texts, labels = create_xy(data_path, 'text', y_column='chart_label_food_insecurity')
	if unstructured_data is not None:
		test_weak_labels(texts, labels)
		df_unstructured, texts_unstructured, labels_unstructured = create_xy(unstructured_data, 'Text')
		X_weak, y_weak = process_weak_labels(texts_unstructured, labels_unstructured)
		
	# print(f'Number data points: {len(df)}')
	true_labels = np.sum(labels)
	# print(f'Number of TRUE labels: {true_labels}')
	# test = cleanData(df) # for TFIDF
	# print(test)

	fig, axs = plt.subplots(1,1, figsize=(10,10))
	# i = 0
	# labs = ['chart']
	# for row in axs:
	# 	for ax in row:
	# label = labels[i]
	data = ' '.join(df.loc[df['chart_label_food_insecurity'] == 0, 'text'])
	cloud = WordCloud(stopwords=STOPWORDS, width=1000, height=1000).generate(data)
	axs.axis('off')
	axs.set_title("Food Insecurity")
	axs.imshow(cloud)

	plt.tight_layout()
	# plt.show()

	plt.savefig("test_neg.png")
 
	# tokens = []
	# for text, label in zip(texts, labels):
	# 	if label == 1:
	# 		# tokens = tokens + word_tokenize(text)
	# 		tokens = tokens + [t for t in word_tokenize(text) if t not in eng_stopwords]
  
	# fdist = FreqDist()

	# for word in tokens: 
	# 	fdist[word.lower()]+= 1

	# print(repr(fdist))
	# print(fdist.most_common(50))
	# weak_label_words = [word[0] for word in fdist.most_common(10)]
	# print(weak_label_words)

	# texts = texts.loc[labels==1] 
	test = pd.concat((texts, labels), axis=1)
	print(test)
	tfIdfVectorizer=TfidfVectorizer(use_idf=True, stop_words=eng_stopwords)
	tfIdf = tfIdfVectorizer.fit_transform(test.loc[test['chart_label_food_insecurity'] == 1, 'text'])
	df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
	df = df.sort_values('TF-IDF', ascending=False)
	print (df.head(50))

	# exit(1)
 
	test_size = 0.2
	seed = 20
	X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, 
                                                    random_state=seed, stratify=labels)

	if unstructured_data is not None:
		X_train, y_train = pd.concat((X_train, X_weak)), pd.concat((y_train, y_weak))

	return X_train, X_test, y_train, y_test

# Clean up each clinical note 
def clean_words(note): 
	note = re.sub(r'[^a-zA-Z\s]', '', note)
	note = re.sub(' +', ' ', note)
	note = re.sub(r'[0-9]','',note)
	words = note.lower()
	# words = words.split()
	return words

# This function takes in a original csv and only returns the text and labels 
def create_xy(data_path, x_column, y_column=None):
	df = pd.read_csv(data_path, nrows=5000)

	# getting rid of N/A labels
	if y_column is not None:
		df = df[df[y_column].notna()]
		assert len(pd.unique(df[y_column])) == 2

	# cleaning clinical notes
	for idx, row in tqdm(df.iterrows()):
		df.at[idx, x_column] = clean_words(row[x_column])	

	texts = df[x_column]

	if y_column is not None:
		# Labeled Data (supervised learning)
		labels = df[y_column].astype(int)
	else:
		# Unlabeled Data (weak labeling)
		labels = pd.Series(np.zeros_like(texts, dtype=int))

	# Get rid of rows that have empty Texts 
	texts  = texts.replace(r'^s*$', float('NaN'), regex = True)
	labels = labels[texts.notna()]
	texts  = texts[texts.notna()]

	assert len(texts) == len(labels)

	return df, texts, labels

def process_weak_labels(texts, labels):
	assert len(texts) == len(labels)
	for idx, text in tqdm(enumerate(texts)):
		intersection = set(text.split(' ')).intersection(weak_label_words)
		if len(intersection) > 0:
			labels.iloc[idx] = 1
	assert len(texts) == len(labels)
	return texts, labels

def test_weak_labels(texts, labels):
	dummy_labels = pd.Series(np.zeros_like(texts), dtype=int)
	text_weak, label_weak = process_weak_labels(texts, dummy_labels)
	num_incorrect = np.sum(np.abs(label_weak - labels))
	num_correct   = len(labels) - num_incorrect
	accuracy = num_correct/len(labels)
	print(f'Num positive weak labels: {np.sum(label_weak)}')
	print(f'Num weak labels correct: {num_correct}')
	print(f'Num total examples: {len(labels)}')
	print(f'Accuracy of weak labels: {accuracy}')

def weighted_sampler(labels):
    classes, count = torch.unique(labels, return_counts=True)
    weight = 1. / count
    weight_lst = [weight[t] for t in labels]
    weights = torch.Tensor(weight_lst)
    return WeightedRandomSampler(weights, num_samples=len(weights))


if __name__ == "__main__":
	# X_train, X_test, y_train, y_test = load_data("../data/food_insecurity_labels_v2.csv", 
	#                                              unstructured_data="../data/unstructured_data.csv")
	# X_train, X_test, y_train, y_test = load_data("../data/food_insecurity_labels_v2.csv", 
	#                                              unstructured_data="../data/raw_weak_labels.csv")
	X_train, X_test, y_train, y_test = load_data("../data/food_insecurity_labels_v2.csv")