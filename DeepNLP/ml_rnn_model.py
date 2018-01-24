import sys
import os
import gensim
import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import rnn
from gensim.models import Word2Vec
from collections import Counter
import pdb

DATA_PATH = "deepnlp/clean_chatbot.txt"
MIN_COUNT = 2
SIZE = 128
WINDOW = 2
VOCAB_SIZE = 30000
MAX_LEN_CBTS = 320
LEARNING_RATE = 1
NUM_TRAINING_STEPS = 400
WEIGHTS_FLD = 'processed/'
SKIP_STEP = 50
BATCH_SIZE = 16
NUM_TRAINING_STEPS = 100
HIDDEN_SIZE = 128
EMBED_SIZE = 128
TIMESTEP = 320
DISPLAY_STEP = 10

def RNN(X, weights, biases):
	X = tf.unstack(X, TIMESTEP, 1)
	lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias = 1.0)
	outputs, states = rnn.static_rnn(lstm_cell, X, dtype = tf.float32)
	return tf.matmul(outputs[-1], weights['out_layer']) + biases['out_layer']

def train_model(index_words, chatbot_output):
	X = tf.placeholder(tf.float32, shape = [None, TIMESTEP, EMBED_SIZE], name = 'X')
	target = tf.placeholder(tf.float32, shape = [None, 2], name = 'target')

	weights = {'out_layer': tf.Variable(tf.random_normal([HIDDEN_SIZE, 2]))}
	biases = {'out_layer' : tf.Variable(tf.random_normal([2]))}

	logits = RNN(X, weights, biases)
	prediction = tf.nn.softmax(logits)

	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = target))
	optimizer = tf.train.AdagradOptimizer(learning_rate = LEARNING_RATE).minimize(loss_op)

	correct_pred = tf.equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.global_variables_initializer()

	cbt_opt = tf.one_hot(chatbot_output, 2, 1, 0)

	with tf.Session() as sess:
		sess.run(init)

		for step in range(1, NUM_TRAINING_STEPS+1):
			batch_x, batch_y = index_words, cbt_opt
			#batch_x = batch_x.reshape(80, TIMESTEP, EMBED_SIZE)
			#print(batch_x.shape)
			#batch_y = cbt_opt
			#batch_x = batch_x.reshape((len(batch_x), TIMESTEP, EMBED_SIZE))
			#batch_x = tf.reshape(batch_x, [len_batch_x, TIMESTEP, EMBED_SIZE])
			sess.run(optimizer, feed_dict = {X: batch_x, target: batch_y})
			if step % DISPLAY_STEP == 0 or 1:
				loss, acc = sess.run([loss_op, accuracy], feed_dict = {X:batch_x, Y:batch_y})
				print("Step" + str(step) + "Minibatch Loss = "+"{.4f}".format(loss) + "Training Accuracy"+
					"{:.3f}".format(acc))
		print("Optimization finished")

def load_word_vectors(framework, word_vec_gensim_model, index_words, chatbot_output):
	if framework == 'Gensim':
		#target = tf.one_hot(chatbot_output, 2, 1.0, 0.0)
		#print(target)
		word_embeddings = []
		for key in word_vec_gensim_model.vocab.keys():
			lst_temp = []
			lst_temp = word_vec_gensim_model[key]
			word_embeddings.append(lst_temp)
		word_embeddings = np.array(word_embeddings)
		#print(word_embeddings.shape)
		#print(word_embeddings.shape[0])
		#embedding_matrix = tf.Variable(tf.constant(0.0, shape = word_embeddings.shape), trainable = False, name = 'embed_matrix')
		#embedding_placeholder = tf.placeholder(tf.float32, word_embeddings.shape)
		#embedding_init = W.assign(embedding_placeholder)
		X = tf.placeholder(tf.float32, shape = [None, TIMESTEP], name = 'X')
		target = tf.placeholder(tf.float32, shape = [None], name = 'target')
		target = tf.one_hot(target, 2, 1, 0)
		
		#word_vectors = tf.contrib.layers.embed_sequence(index_words, vocab_size = word_embeddings.shape[0],
			#embed_dim = word_embeddings.shape[1])
		X = tf.contrib.layers.embed_sequence(X, vocab_size = word_embeddings.shape[0], embed_dim = word_embeddings.shape[1])

		#print(word_vectors.shape)
		#word_list = tf.unstack(word_vectors, axis = 1)
		X = tf.unstack(X, axis = 1)
		#print(word_list)
		cell = tf.contrib.rnn.GRUCell(word_embeddings.shape[1])
		#_, encoding = tf.contrib.rnn(cell, word_list, dtype = tf.float32)
		#_, encoding = rnn.static_rnn(cell, word_list, dtype = tf.float32)
		_, encoding = rnn.static_rnn(cell, X, dtype =tf.float32)

		logits = tf.contrib.layers.fully_connected(encoding, 2, activation_fn = None)
		loss = tf.losses.softmax_cross_entropy(logits, target)

		global_step = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')

		train_op = tf.contrib.layers.optimize_loss(loss, global_step = global_step, optimizer = 'Adam', learning_rate = 0.01, clip_gradients = 1.0)

		init = tf.global_variables_initializer()

		with tf.Session as sess:
			sess.run(init)
			for step in range(1, NUM_TRAINING_STEPS+1):
				sess.run(train_op, feed_dict = {X : index_words, target : cbt_opt})
				if step % DISPLAY_STEP == 0 or 1:
					loss = sess.run(loss, feed_dict = {X : index_words, target : cbt_opt})
				print("Step" + str(step) + "Minibatch Loss = "+"{.4f}".format(loss))
		print("Optimization finished")
		
def embed_vectors_input(index_words, word_vec_gensim_model):
	idx_words_embed = []
	word_embeddings = []
	for key in word_vec_gensim_model.vocab.keys():
		lst_temp = []
		lst_temp = word_vec_gensim_model[key]
		word_embeddings.append(lst_temp)
	word_embeddings = np.array(word_embeddings)
	idx_words_embed = tf.contrib.layers.embed_sequence(index_words, vocab_size = word_embeddings.shape[0],
		embed_dim = word_embeddings.shape[1])
	print(idx_words_embed)
	sess = tf.Session()
	print(sess.run(idx_words_embed))

	#idx_words_embeds = tf.unstack(idx_words_embed, TIMESTEP, axis = 1)
	#print(len(idx_words_embeder))
	#print(idx_words_embeds)
	#idx_words_embeder = tf.stack(idx_words_embeds, axis = 1)
	#sess = tf.Session()
	#with sess.as_default():
		#idx_words_embeder = (sess.run(idx_words_embeder))
	#print(idx_words_embeder)
	#print(type(idx_words_embeder))
	idx_words_embeder = np.array(idx_words_embeds)
	print(idx_words_embeder)
	return idx_words_embeder

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
    lst_idx_opts = []
    for word_lst in lst_sqs:
    	if (len(word_lst) != 320):
    		word_lst.extend([0] * (320 - len(word_lst)))
    	lst_idx_opts.append(word_lst)
    return lst_idx_opts

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

def load_ouput(file_path):
	output_lst = []
	with open(file_path, encoding = 'utf-8') as f:
		for line in f:
			output_lst.append(str(line))
	return output_lst

if __name__ == '__main__':
	words, gen_words = get_words_from_txt(DATA_PATH)
	word_dict, index_dict = build_vocab(words, VOCAB_SIZE)
	index_words = convert_words_to_index(gen_words, index_dict)
	# Load outputs from the chatbot file
	chatbot_output = load_ouput("deepnlp/chatbot_output.txt")
	print(len(index_words))
	del words
	word_vec_gensim_model =  build_gensim_word_vec_model(index_words, MIN_COUNT, SIZE, WINDOW)
	#print(words)
	#model = Word2Vec(words, min_count = MIN_COUNT, size = SIZE, window = WINDOW)
	word_vocab = list(word_vec_gensim_model.vocab.keys())
	#print(len(word_vocab))
	#print(word_vec_gensim_model[1])
	chatbot_output_ctr = []
	for flag in chatbot_output:
		if flag == 'flagged':
			chatbot_output_ctr.append(1)
		else:
			chatbot_output_ctr.append(0)
	load_word_vectors('Gensim', word_vec_gensim_model, index_words, chatbot_output_ctr)
	#idx_words_embed = embed_vectors_input(index_words, word_vec_gensim_model)
	#train_model(idx_words_embed, chatbot_output_ctr)