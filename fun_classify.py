"""

Author: Ruixian Zhao

 This fun classify script will launch the best model for IMDB Review dataset.
 And take the input of sentence (reviews from website).
 And judge the sentence mood.

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

from helper1 import get_dataset, prepare_word_dic
from imdb_launcher import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re, string
regex = re.compile('[%s]' % re.escape(string.punctuation))

data=get_dataset('imdb')
word_index = data.get_word_index()
word_index = prepare_word_dic(word_index)


raw_text = 'This is movie is very fun, I like it.'
print(raw_text)
raw_text = regex.sub('', raw_text)
raw_text = raw_text.lower()
test_word=raw_text.split(' ')
test_code=[]

for i in test_word:
	a = int(word_index.get(i, 0))
	test_code = np.concatenate((test_code,[a])) 
	test_code = np.append(test_code,a)


test_code = np.asarray([test_code])

test_data = pad_sequences(test_code, value=word_index["<PAD>"], padding='post', maxlen=256)

model = new_model

predict = model.predict_classes(test_data)


if predict == 1:
	print('This review is positive!')
else:
	print('This reveiw is negetive!')

