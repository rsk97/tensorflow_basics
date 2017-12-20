import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
nm_lines = 10000000

def make_lexicon(pos,neg):
	lexicon = []
	for file in [pos,neg]:
		with open(file,'r') as f:
			contents = f.readlines()
			for l in contents[:nm_lines]:
				all_words = word_tokenize(l)
				lexicon+=list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_count = Counter(lexicon)
	l = []

	for word in w_count:
		if(1000 > w_count[word] > 50):
			l.append(word)

	return l

def sample_handling(sample,lexicon,classification):
	featureset = []

	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:nm_lines]:
			curr_word = word_tokenize(l.lower())
			curr_word = [lemmatizer.lemmatize(i) for i in curr_word]
			feat = np.zeros(len(lexicon))
			for word in curr_word:
				if(word.lower() in lexicon):
					in_val = lexicon.index(word.lower())
					feat[in_val] += 1
			feat = list(feat)
			featureset.append([feat,classification])

	return featureset

def create_feature_label(pos,neg,test_size = 0.1):
	lexicon = make_lexicon(pos,neg)
	feat = []
	feat += sample_handling('pos.txt',lexicon,[1,0])
	feat += sample_handling('neg.txt',lexicon,[0,1])
	random.shuffle(feat)
	feat =np.array(feat)
	testing_size = int(test_size*len(feat))

	train_x = list(feat[:,0][:-testing_size])
	train_y = list(feat[:,1][:-testing_size])

	test_x = list(feat[:,0][-testing_size:])
	test_y = list(feat[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y

if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_label('pos.txt','neg.txt')

	with open('sentimentset.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
