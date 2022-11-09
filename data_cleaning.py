import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_data(data_path):
	df = pd.read_csv(data_path)
	import pdb;pdb.set_trace()

if __name__ == "__main__":
	x, y = load_data("data/food_insecurity_labels_v2.csv")