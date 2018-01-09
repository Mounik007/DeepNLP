from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import re
import tensorflow as tf
import numpy as np
import random
#import pdb

from six.moves import urllib
from collections import Counter

VOCAB_SIZE = 30000
DATA_PATH = "deepnlp/clean_chatbot.txt"
EMBED_SIZE = 128
SKIP_WINDOW = 1
NUM_SAMPLED = 64
LEARNING_RATE = 1.0
NUM_TRAINING_STEPS = 100000
WEIGHTS_FLD = 'processed/'
SKIP_STEP = 2000
BATCH_SIZE = 16

class SkipGramModel:
	def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.batch_size = batch_size
		self.num_sampled = num_sampled
		self.learning_rate =  learning_rate
		self.global_step = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')

	def create_placeholders(self):
		""" Define the place holders for input and output """
		with tf.name_scope("data"):
			self.center_words = tf.placeholder(tf.int32, shape = [self.batch_size], name = 'center_words')
			self.target_words = tf.placeholder(tf.int32, shape = [self.batch_size, 1], name = 'target_words')

	def create_embedding(self):
		""" Define weights"""
		with tf.device('/cpu:0'):
			with tf.name_scope("embed"):
				self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size],
					-1.0, 1.0), name = 'embed_matrix')

	def create_loss(self):
		''' Define loss and model '''
		with tf.device('/cpu:0'):
			with tf.name_scope("loss"):
				embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name ='embed')
				
				nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
					stddev = 1.0/(self.embed_size ** 0.5)), name = 'nce_weight')
				
				nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name = 'nce_bias')
				
				self.loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weight, biases = nce_bias,
					labels = self.target_words, inputs = embed, num_sampled = self.num_sampled,
					num_classes = self.vocab_size), name = 'loss')

	def create_optimizer(self):
		""" Define optimizer"""
		with tf.device('/cpu:0'):
			self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step = self.global_step)

	def create_summaries(self):
		with tf.name_scope("summaries"):
			tf.summary.scalar("loss", self.loss)
			#tf.summary.histogram("histogram loss", self.loss)
			self.summary_op = tf.summary.merge_all()

	def build_graph(self):
		self.create_placeholders()
		self.create_embedding()
		self.create_loss()
		self.create_optimizer()
		self.create_summaries()

def genereate_sample(index_words, context_window_size):
	""" Form training pairs according to the skip gram model"""
	for idx, center in enumerate(index_words):
		ctx = random.randint(1, context_window_size)
		for tgt in index_words[max(0, idx - ctx) : idx]:
			yield center, tgt
		for target in index_words[idx + 1: idx + ctx + 1]:
			yield center, target

def get_batch(iterator, batch_size):
	while True:
		center_batch = np.zeros(batch_size, dtype = np.int32)
		target_batch = np.zeros([batch_size, 1], dtype = np.int32)
		for idx in range(batch_size):
			#pdb.set_trace()
			center_batch[idx], target_batch[idx] = next(iterator)
		yield center_batch, target_batch

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def get_words_from_txt(data_path):
	words = []
	with open(data_path, encoding = 'utf-8') as f:
		for line in f:
			for word in line.split():
				words.append(word.lower())
	return words

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

def train_model(model, batch_gen, num_train_steps, weights_fld):
	saver = tf.train.Saver()
	initial_step = 0
	counter = 1
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		total_loss = 0
		writer = tf.summary.FileWriter('imporved_graph/lr' + str(LEARNING_RATE), sess.graph)
		initial_step = model.global_step.eval()
		for idx in range(initial_step, initial_step + num_train_steps):
			#pdb.set_trace()
			counter += 1
			print(counter)
			centers, targets = next(batch_gen)
			feed_dict = {model.center_words: centers, model.target_words: targets}
			loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict = feed_dict)
			writer.add_summary(summary, global_step = idx)
			total_loss += loss_batch
			if(idx + 1) % SKIP_STEP == 0:
				print('Average loss at step {}: {:5.1f}'.format(idx, total_loss/SKIP_STEP))
				total_loss = 0
				saver.save(sess, 'checkpoints/skip-gram', idx)

def start_train_process(batch_gen):
	model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
	model.build_graph()
	#batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
	train_model(model, batch_gen, NUM_TRAINING_STEPS, WEIGHTS_FLD)

def generate_batch(index_words):
	single_gen = genereate_sample(index_words, SKIP_WINDOW)
	return get_batch(single_gen, BATCH_SIZE)

if __name__ == '__main__':
	words = get_words_from_txt(DATA_PATH)
	print(len(words))
	#print(VOCAB_SIZE)
	word_dict, index_dict = build_vocab(words, VOCAB_SIZE)
	index_words = convert_words_to_index(words, index_dict)
	del words # To save space
	#pdb.set_trace()
	batch_gen = generate_batch(index_words)
	start_train_process(batch_gen)
	print("Completed")
