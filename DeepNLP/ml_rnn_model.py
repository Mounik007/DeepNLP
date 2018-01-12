import sys
import os
import gensim
import numpy as np
import random
from gensim.models import Word2Vec
#from genereate_word_vec import get_words_from_txt

DATA_PATH = "deepnlp/clean_chatbot.txt"
MIN_COUNT = 2
SIZE = 128
WINDOW = 2

def get_words_from_txt(data_path):
	words = []
	with open(data_path, encoding = 'utf-8') as f:
		for line in f:
			temp_list = []
			for word in line.split():
				temp_list.append(word.lower())
			words.append(temp_list)
	return words

if __name__ == '__main__':
	words = get_words_from_txt(DATA_PATH)
	#print(words)
	model = Word2Vec(words, min_count = MIN_COUNT, size = SIZE, window = WINDOW)
	word_vocab = list(model.wv.vocab.keys())
	print(model['little'])