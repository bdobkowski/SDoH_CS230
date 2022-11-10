import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import re
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

def load_data(data_path):
	df = pd.read_csv(data_path)

	# print(df.columns)

	# getting rid of N/A labels
	df = df[df['chart_label_food_insecurity'].notna()]
	assert len(pd.unique(df['chart_label_food_insecurity'])) == 2

	# print(df['text'][0][0:150])

	# cleaning clinical notes
	for idx, row in df.iterrows():
		df.at[idx, 'text'] = clean_words(row['text'])

	# print(df['text'][0][0:150])

	# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# out = tokenizer.tokenize(df['text'][0])

	# print(f'Number data points: {len(df)}')
	true_labels = np.sum(df['chart_label_food_insecurity'])
	# print(f'Number of TRUE labels: {true_labels}')
	# import pdb;pdb.set_trace()
	texts = df['text']
	labels = df['chart_label_food_insecurity'].astype(int)
	test_size = 0.2
	seed = 20
	X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, 
                                                    random_state=seed, stratify=labels)
	return X_train, X_test, y_train, y_test

# Clean up each clinical note 
def clean_words(note): 
	note = re.sub(r'[^a-zA-Z\s]', '', note)
	note = re.sub(' +', ' ', note)
	note = re.sub(r'[0-9]','',note)
	words = note.lower()
	# words = words.split()
	return words


def weighted_sampler(labels):
    classes, count = torch.unique(labels, return_counts=True)
    weight = 1. / count
    weight_lst = [weight[t] for t in labels]
    weights = torch.Tensor(weight_lst)
    return WeightedRandomSampler(weights, num_samples=len(weights))


if __name__ == "__main__":
	X_train, X_test, y_train, y_test = load_data("../data/food_insecurity_labels_v2.csv")