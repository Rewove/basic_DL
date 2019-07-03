"""

Author: Ruixian Zhao

 This gridsearch script contains all the attampts among the adjusted on model 2 in IMDB dataset.


"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# prepare the dataset
from helper1 import get_dataset, prepare_imdb
# prepare functions
from helper2 import *
from helper3 import *
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Flatten
vocab_size = 10000
seed = 7
np.random.seed(seed)
from keras.models import Sequential
from keras import regularizers
from keras.layers import Embedding, Conv1D, Dense, Flatten, MaxPooling1D, Dropout
vocab_size = 10000
max_words = 256
seed = 7
np.random.seed(seed)
(train_data, train_labels), (test_data, test_labels), (x_val, partial_x_train), (y_val, partial_y_train) = prepare_imdb(dataset)

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
maxnorm
'''
def create_model():
    model = Sequential()
    model.add(Embedding(vocab_size, 32 , input_length=max_words))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
    model = Sequential()
    model.add(Embedding(vocab_size, 16))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation=tf.nn.relu))
    model.add(Dense(1, activation=tf.nn.sigmoid))

learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)


optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']



init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)


activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)

dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate)

'''
a=0
b=1

from keras.optimizers import Adamax

if b:
    def create_model():
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    batch_size = [256, 512] 
    epochs = [5, 10, 15] 
    # Best: 0.865600 using {'batch_size': 256, 'epochs': 5}
    batch_size = [64, 128, 256, 512] 
    epochs = [5, 8, 10, 15] 
    param_grid = dict(batch_size=batch_size, epochs=epochs) 
    # Best: 0.866000 using {'batch_size': 128, 'epochs': 5}
    model = KerasClassifier(build_fn=create_model, verbose=0)
    
    def create_model(optimizer='Adam'):
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer) 
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
    #Best: 0.866267 using {'optimizer': 'Adamax'}
    

if a:
    def create_model(learn_rate=0.001):
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adamax(lr=learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    learn_rate = [0.001, 0.002, 0.005, 0.01,0.02]
    learn_rate = [0.0015, 0.002, 0.0025, 0.003]
    param_grid = dict(learn_rate=learn_rate)
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
    # Best: 0.863733 using {'learn_rate': 0.01} 0.1,2,3
    # Best: 0.871867 using {'learn_rate': 0.002}
    # Best: 0.870800 using {'learn_rate': 0.0015}
    
    
if a:
    def create_model(init_mode='uniform'):
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(16, kernel_initializer=init_mode, activation='relu'))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    param_grid = dict(init_mode=init_mode)
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
    # Best: 0.873000 using {'init_mode': 'uniform'}
    # Best: 0.870733 using {'init_mode': 'glorot_normal'}
    

if a:
    def create_model(activation='relu'):
        init_mode='glorot_normal'
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(16, kernel_initializer=init_mode, activation=activation))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(activation=activation)
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
    # Best: 0.871800 using {'activation': 'tanh'}
    # Best: 0.870867 using {'activation': 'sigmoid'}

if a: 
    def create_model(neurons=1):
        init_mode='uniform'
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(neurons, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    neurons = [1, 5, 10, 16, 32, 64, 128, 256, 512]
    param_grid = dict(neurons=neurons)
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
    # Best: 0.871800 using {'neurons': 16}
    ######################################

if a:  # too time consume
    def create_model(neurons1=8, neurons2=8):
        init_mode='uniform'
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(neurons1, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(neurons2, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    # multi parameters with double layer
    neurons1 = [8, 16, 32, 64, 128, 256, 512]
    neurons2 = [8, 16, 32, 64, 128, 256, 512]
    epochs = [5, 8, 10, 15] 
    batch_size = [64, 128, 256, 512] 
    param_grid = dict(neurons1=neurons1, neurons2=neurons2,
                      batch_size = batch_size, epochs=epochs)
    model = KerasClassifier(build_fn=create_model, verbose=0)
    print("two nuroals and epochs and batch size with double layer")
    # ???

if a: 
    def create_model(neurons=1):
        init_mode='uniform'
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(neurons, kernel_initializer=init_mode, activation='tanh'))
        # model.add(Dense(neurons, kernel_initializer=init_mode, activation=activation))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    #singel more epochs 15
    neurons = [1, 5, 10, 16, 32, 64, 128, 256, 512]
    param_grid = dict(neurons=neurons)
    model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=128, verbose=0)
    # Best: 0.862333 using {'neurons': 10}


if a: 
    def create_model(neurons=1):
        init_mode='uniform'
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(neurons, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(neurons, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    # double layer with more epoches
    neurons = [1, 5, 10, 16, 32, 64, 128, 256, 512]
    param_grid = dict(neurons=neurons)
    model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=128, verbose=0)
    # epoch 15 is not good
    # Best: 0.863200 using {'neurons': 5}

if a: 
    def create_model(neurons1=1,neurons2=1):
        init_mode='uniform'
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(neurons1, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(neurons2, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    # double layers with good epoches but with different combination of nureals
    neurons1 = [1,2,3,4,5,6,7,8,16]
    neurons2 = [1,2,3,4,5,6,7,8,16]
    param_grid = dict(neurons1=neurons1, neurons2 = neurons2)
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
   	# Best: 0.870733 using {'neurons1': 4, 'neurons2': 16}


if a: 
    def create_model(neurons1=1):
        init_mode='uniform'
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(neurons1, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    # single layers with good epoches 
    # next with maxpooling
    #neurons1 = [1,2,3,4,5,6,7,8,16]
    #neurons1 = [3,16,128]
    neurons1 = [13,14,15,16,17,18,19,20]
    epochs = [3,4,5,6,7,8]
    param_grid = dict(neurons1=neurons1,epochs=epochs)
    model = KerasClassifier(build_fn=create_model, batch_size=128, verbose=0)
    testing = 'single_?e_neurons'
   	# Best: 0.868400 using {'neurons1': 3}
   	# Best: 0.868800 using {'neurons1': 16}
   	# change the epoches 
   	# Best: 0.870333 using {'epochs': 4, 'neurons1': 16}


if a: 
    def create_model(neurons1=16,neurons2=16):
        init_mode='uniform'
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(neurons1, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(neurons2, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    #neurons1 = [14,16,18,20]
    #epochs = [3,4,5,6,7]
    neurons1 = [10,20,24,32]
    neurons2 = [10,20,24,32]
    epochs = [4,5]
    neurons1 = [32,64,128]
    neurons2 = [24,32]
    epochs = [5,6]
    param_grid = dict(neurons1=neurons1, neurons2=neurons2, epochs=epochs)
    model = KerasClassifier(build_fn=create_model, batch_size=128, verbose=0)
    testing = 'max_single_?e_?neurons'
    testing = 'max_double_?e_?n'
   	# Best: 0.872867 using {'epochs': 4, 'neurons1': 20}
   	########################################################################
   	# Best: 0.870333 using {'epochs': 4, 'neurons1': 10} # double neurons2 =16
   	# Best: 0.871333 using {'epochs': 5, 'neurons1': 32, 'neurons2': 24}
   	# Best: 0.866533 using {'epochs': 5, 'neurons1': 32, 'neurons2': 32}
   	# fine only one


from keras.constraints import maxnorm

if a:
    def create_model(dropout_rate=0.0, weight_constraint=1):
        init_mode='uniform'
        activation='tanh'
        neurons1 = 250
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(neurons1, kernel_initializer=init_mode, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0,0.2,0.4,0.6,0.8]
    param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
    testing = 'drop_large_hidden'
    # Best: 0.871667 using {'dropout_rate': 0.8, 'weight_constraint': 1}


if a:
    def create_model(dropout_rate=0.0, weight_constraint=0):
        init_mode='uniform'
        activation='tanh'
        neurons1 = 4
        neurons2 = 16
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(neurons1, kernel_initializer=init_mode, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons2, kernel_initializer=init_mode, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0,0.2,0.4,0.6,0.8]
    param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
    testing = 'drop_samll_hidden'
    # Best: 0.870867 using {'dropout_rate': 0.0, 'weight_constraint': 3}

if a:
    def create_model(init_mode='uniform'):
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(20, kernel_initializer=init_mode, activation='tanh'))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    param_grid = dict(init_mode=init_mode)
    model = KerasClassifier(build_fn=create_model, epochs=4, batch_size=128, verbose=0)
    # Best: 0.873000 using {'init_mode': 'uniform'}
    # tanh epoch 4 nuro 20
    # Best: 0.874333 using {'init_mode': 'lecun_uniform'}
    # relu


if a:

	def create_model(activation1='relu', activation2='relu'):
		model = Sequential()
		model.add(Embedding(vocab_size, 32 , input_length=max_words))
		model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation=activation1))
		model.add(MaxPooling1D(pool_size=2))
		model.add(Flatten())
		model.add(Dense(20, kernel_initializer='lecun_uniform', activation=activation2))
		model.add(Dense(1, kernel_initializer='lecun_uniform', activation='sigmoid'))
		optimizer = Adamax(lr=0.002)
		model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return model
    
	activation1 = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
	activation2 = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']    
	param_grid = dict(activation1=activation1, activation2=activation2)
	model = KerasClassifier(build_fn=create_model, epochs=4, batch_size=128, verbose=0)
	# Best: 0.874200 using {'activation1': 'relu', 'activation2': 'softsign'}
	# average pooling


if a:
    def create_model(activation2='softsign'):
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(GlobalAveragePooling1D())
        #model.add(Flatten())
        model.add(Dense(20, kernel_initializer='lecun_uniform', activation=activation2))
        model.add(Dense(1, kernel_initializer='lecun_uniform', activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    activation2 = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'] 
    param_grid = dict(activation2=activation2)
    model = KerasClassifier(build_fn=create_model, epochs=4, batch_size=128, verbose=0)
    # Best: 0.866867 using {'activation2': 'linear'}
    # not good


if a:
    def create_model(dropout_rate=0.0, weight_constraint=0):
        init_mode='lecun_uniform'
        activation='softsign'
        neurons1 = 20
        model = Sequential()
        model.add(Embedding(vocab_size, 32 , input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(neurons1, kernel_initializer=init_mode, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0,0.2,0.4,0.6,0.8]
    param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
    testing = 'drop_best'
    # Best: 0.876200 using {'dropout_rate': 0.2, 'weight_constraint': 2}





grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1) 
grid_result = grid.fit (partial_x_train, partial_y_train) 
# summarize results 
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
