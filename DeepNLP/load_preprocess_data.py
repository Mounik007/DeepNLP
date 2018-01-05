 #!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import csv
import re
import collections
import pdb

NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE) # match all whitespace except newlines
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(chr(768), chr(769), chr(832),
                                                                                      chr(833), chr(2387), chr(5151),
                                                                                      chr(5152), chr(65344), chr(8242)),
                                  re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
ALLOWED_CURRENCIES = """¥£₪$€฿₨"""
ALLOWED_PUNCTUATION = """-!?/;"'%&<>.()[]{}@#:,|=*"""
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}{}]'.format(re.escape(ALLOWED_CURRENCIES), re.escape(ALLOWED_PUNCTUATION)), re.UNICODE)

def write_clean_text(responses, file_name):
	with open(file_name, 'w', encoding = 'latin1') as file_handler:
		for response in responses:
			file_handler.write("{}\n".format(response))
	print("Done writting to the file")


def count_max_words(comments):
	max_word_count = 0
	for comment in comments:
		word_list = []
		word_list = comment.split(' ')
		if(len(word_list) > max_word_count):
			max_word_count = int(len(word_list))
	return max_word_count

def count_highest_characters_string(comments):
	highest_counter = 0
	for comment in comments:
		if(len(comment) > highest_counter):
			highest_counter = int(len(comment))
	return highest_counter

def clean_text(text):
	"""Clean the text - remove unwanted chars, fold punctuation etc."""
	result = NORMALIZE_WHITESPACE_REGEX.sub(' ', text.strip())
	result = RE_DASH_FILTER.sub('-', result)
	result = RE_APOSTROPHE_FILTER.sub("'", result)
	result = RE_LEFT_PARENTH_FILTER.sub("(", result)
	result = RE_RIGHT_PARENTH_FILTER.sub(")", result)
	result = RE_BASIC_CLEANER.sub('', result)
	return result

def preprocess_text(response_text):
	response_clean_text = []
	for text in response_text:
		#print(text)
		#decoded_text = text.decode('utf-8')
		cleaned_text = clean_text(text)
		cleaned_text = re.sub("\s\s+", " ", cleaned_text)
		response_clean_text.append(cleaned_text)
	return response_clean_text

def load_data(file_path):
	row_id = []
	class_flag = []
	response_text = []
	with open(file_path, encoding = 'latin1') as csv_file:
		read_csv = csv.reader(csv_file, delimiter = ',')
		next(read_csv)
		for row in read_csv:
			row_id.append(row[0])
			class_flag.append(row[1])
			response_text.append(row[2])
	return row_id, class_flag, response_text

if __name__ == '__main__':
	file_chatbot_path = 'deepnlp/Sheet_1.csv'
	row_id_chatbot, class_flag_chatbot, response_text_chatbot = load_data(file_chatbot_path)
	#print(response_text_chatbot)
	response_clean_text_chatbot = preprocess_text(response_text_chatbot)
	#print(response_clean_text_chatbot)
	file_resume_path = 'deepnlp/Sheet_2.csv'
	#pdb.set_trace()
	row_id_resume, class_flag_resume, response_text_resume = load_data(file_resume_path)
	print(len(response_text_resume))
	response_clean_text_resume = preprocess_text(response_text_resume)
	print(len(response_clean_text_resume))
	write_clean_text(response_clean_text_chatbot, "deepnlp/clean_chatbot.txt")
	write_clean_text(response_clean_text_resume, "deepnlp/clean_resume.txt")
	# Analysis of the texts
	max_char_count_chatbot = 0
	max_char_count_chatbot = count_highest_characters_string(response_clean_text_chatbot)
	max_word_count_chatbot = 0
	max_word_count_chatbot = count_max_words(response_clean_text_chatbot)
	max_char_count_resume = 0
	max_char_count_resume = count_highest_characters_string(response_clean_text_resume)
	max_word_count_resume = 0
	max_word_count_resume = count_max_words(response_clean_text_resume)
	print(80 * "=")
	print("\n")
	print("Maximum characters in the chatbot set:\n")
	print(max_char_count_chatbot)
	print("\n")
	print(80 * "=")
	print("\n")
	print("Maximum words in the chatbot set: \n")
	print(max_word_count_chatbot)
	print(80 * "=")
	print("\n")
	print("Maximum characters in the resume set:\n")
	print(max_char_count_resume)
	print("\n")
	print(80 * "=")
	print("\n")
	print("Maximum words in the resume set: \n")
	print(max_word_count_resume)