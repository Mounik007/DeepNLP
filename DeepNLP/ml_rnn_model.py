import sys
import os
import gensim
import numpy as np
import random
from gensim.models import Word2Vec
from collections import Counter

DATA_PATH = "deepnlp/clean_chatbot.txt"
MIN_COUNT = 2
SIZE = 128
WINDOW = 2
VOCAB_SIZE = 30000
MAX_LEN_CBTS = 320

def convert_words_to_index(words_seq, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    lst_sqs = []
    for words in words_seq:
    	temp_list = []
    	for word in words:
    		if word in dictionary:
    			temp_list.append(dictionary[word])
    		else:
    			temp_list.append(0)
    	lst_sqs.append(temp_list)
    return lst_sqs

    #return [dictionary[word] if word in dictionary else 0 for word in words]

def build_vocab(words, vocab_size):
	word_dict = [('UNK', -1)]
	word_dict.extend(Counter(words).most_common(vocab_size - 1))
	index_dict = {}
	index = 0
	with open('vocab_1000.tsv', 'w') as f:
		for key, value in word_dict:
			index_dict[key] = index
			if index < 30000:
				f.write(key + "\t" +str(index) +"\n")
			index += 1
	return word_dict, index_dict

def build_gensim_word_vec_model(words, min_count, size, window):
	model = Word2Vec(words, min_count = min_count, size = size, window = window)
	return model

def get_words_from_txt(data_path):
	words = []
	gen_words = []
	with open(data_path, encoding = 'utf-8') as f:
		for line in f:
			temp_list = []
			for word in line.split():
				words.append(word.lower())
				temp_list.append(word.lower())
			gen_words.append(temp_list)
	return words, gen_words

if __name__ == '__main__':
	words, gen_words = get_words_from_txt(DATA_PATH)
	word_dict, index_dict = build_vocab(words, VOCAB_SIZE)
	index_words = convert_words_to_index(gen_words, index_dict)
	print(len(index_words))
	del words
	word_vec_gensim_model =  build_gensim_word_vec_model(index_words, MIN_COUNT, SIZE, WINDOW)
	#print(words)
	#model = Word2Vec(words, min_count = MIN_COUNT, size = SIZE, window = WINDOW)
	word_vocab = list(word_vec_gensim_model.vocab.keys())
	print(len(word_vocab))
	print(word_vec_gensim_model[1])