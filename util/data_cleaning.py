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

# y_column_name = 'chart_label_food_insecurity'
y_column_name = 'outcomes'

# these are the top n words after the stopwords in the positive labeled data
# weak_label_words = ['mg', 'patient', 'pain', 'daily', 'history', 'pt', 'md', 'labs', 'per', 'tablet']

def load_data(data_path, unstructured_data=None, weak_label_words=None):
    
	# df, texts, labels = create_xy(data_path, 'text', y_column='chart_label_food_insecurity')
	df, texts, labels = create_xy(data_path, 'text', y_column=y_column_name)
	if unstructured_data is not None: 
		df_unstructured, texts_unstructured, labels_unstructured = create_xy(unstructured_data, 'Text')
		X_weak, y_weak = process_weak_labels(texts_unstructured, labels_unstructured, weak_label_words)
 
	test_size = 0.2
	seed = 20
	X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, 
                                                    random_state=seed, stratify=labels)

	if unstructured_data is not None:
		X_train, y_train = pd.concat((X_train, X_weak)), pd.concat((y_train, y_weak))

	return X_train, X_test, y_train, y_test

def load_data_test(data_path, unstructured_data=None, weak_label_words=None, stopwords=STOPWORDS):
    
	# df, texts, labels = create_xy(data_path, 'text', y_column='chart_label_food_insecurity')
	df, texts, labels = create_xy(data_path, 'text', y_column=y_column_name)
	if unstructured_data is not None: 
		test_weak_labels(texts, labels, weak_label_words)
		df_unstructured, texts_unstructured, labels_unstructured = create_xy(unstructured_data, 'Text')
		X_weak, y_weak = process_weak_labels(texts_unstructured, labels_unstructured, weak_label_words)
		
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
	data = ' '.join(df.loc[df[y_column_name] == 0, 'text'])
	cloud = WordCloud(stopwords=stopwords, width=1000, height=1000).generate(data)
	axs.axis('off')
	axs.set_title("Food Insecurity Word Cloud - Negative")
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
	# print(test)
 
	# TF-IDF 
	tfIdfVectorizer=TfidfVectorizer(use_idf=True, stop_words=eng_stopwords)
 
 	# Save top positive words 
	tfIdf_pos = tfIdfVectorizer.fit_transform(test.loc[test[y_column_name] == 1, 'text'])
	df_pos = pd.DataFrame(tfIdf_pos[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
	df_pos = df_pos.sort_values('TF-IDF', ascending=False) 
	# print(df_pos)
	pos_words = list(df_pos.index) 
 
	# Save top negative words 
	tfIdf_neg = tfIdfVectorizer.fit_transform(test.loc[test[y_column_name] == 0, 'text'])
	df_neg = pd.DataFrame(tfIdf_neg[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
	df_neg = df_neg.sort_values('TF-IDF', ascending=False) 
	neg_words = list(df_neg.index)
	
	# Delete words that are popular in both food insecure/ food secure patients 
	pred_words = [value for value in pos_words if value not in neg_words]
	# print(pred_words) 
	# print (df.head(50)) 

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
	df = pd.read_csv(data_path, nrows=1000)

	# getting rid of N/A labels
	if y_column is not None:
		df = df[df[y_column].notna()]
		assert len(pd.unique(df[y_column])) == 2

	# cleaning clinical notes
	for idx, row in df.iterrows():
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

def process_weak_labels(texts, labels, weak_label_words):
	assert len(texts) == len(labels)
	for idx, text in enumerate(texts):
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
	# X_train, X_test, y_train, y_test = load_data("../data/final_foodinsecurity_data.csv") 
	
	nltk.download('punkt')
	nltk.download('stopwords')

	eng_stopwords = stopwords.words('english')
	add_words = ['reports', 'patient', 'plan', 'pt', 'day', 'report']

	for w in add_words:
		eng_stopwords.append(w)

	# weak_label_words = ['hunger','hungry','nutrition','food','skinny','eat','eating']
	# weak_label_words = ['nutrition','food','weight','homeless']
	# According to TF-IDF 
	weak_label_words = ['bacs','wesley','melanoma','burned', 'prison']
	
	# X_train, X_test, y_train, y_test = load_data_test("../data/final_foodinsecurity_data.csv",
	# unstructured_data="../data/raw_weak_labels.csv",
	# weak_label_words=weak_label_words) 
	X_train, X_test, y_train, y_test = load_data_test("../data/final_foodinsecurity_data.csv", stopwords=eng_stopwords)
	
 