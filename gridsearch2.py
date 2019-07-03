"""

Author: Ruixian Zhao

 This gridsearch script 2 contains all the attampts among the adjusted on model 1 in IMDB dataset.


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

from keras.optimizers import Adamax, RMSprop

if b:
    def create_model():
        model = Sequential()
        model.add(Embedding(vocab_size, 16))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(16, activation=tf.nn.relu))
        model.add(Dense(1, activation=tf.nn.sigmoid))
        optimizer = Adamax(lr=0.002)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    batch_size = [128, 256, 512] 
    epochs = [5, 10, 15, 20] 
    # Best: 0.874800 using {'batch_size': 128, 'epochs': 20}
    batch_size = [32, 64, 100, 128] 
    epochs = [18, 22, 24] 
    #Best: 0.877800 using {'batch_size': 100, 'epochs': 24}
    batch_size = [90, 100, 110] 
    epochs = [24, 25, 26, 27] 
    # Best: 0.877533 using {'batch_size': 110, 'epochs': 24}
    param_grid = dict(batch_size=batch_size, epochs=epochs) 
    model = KerasClassifier(build_fn=create_model, verbose=0)

epochs=24
batch_size=100

if a:
    def create_model(optimizer='Adam'):
        model = Sequential()
        model.add(Embedding(vocab_size, 16))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(16, activation=tf.nn.relu))
        model.add(Dense(1, activation=tf.nn.sigmoid))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer) 
    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
# Best: 0.878000 using {'optimizer': 'RMSprop'}
    

if a:
    def create_model(learn_rate=0.001):
        model = Sequential()
        model.add(Embedding(vocab_size, 16))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(16, activation=tf.nn.relu))
        model.add(Dense(1, activation=tf.nn.sigmoid))
        optimizer = RMSprop(lr=learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    learn_rate = [0.001, 0.002, 0.005, 0.01,0.02]
    learn_rate = [0.0015, 0.0005, 0.001]
    param_grid = dict(learn_rate=learn_rate)
    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
    # Best: 0.878133 using {'learn_rate': 0.001}
    #Best: 0.876733 using {'learn_rate': 0.0005}
learn_rate=0.001
    
if a:
    def create_model(init_mode='uniform'):
        model = Sequential()
        model.add(Embedding(vocab_size, 16))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(16, kernel_initializer=init_mode, activation='relu'))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = RMSprop(lr=learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    param_grid = dict(init_mode=init_mode)
    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
    # Best: 0.878867 using {'init_mode': 'he_normal'}

init_mode='he_normal'

if a:
    def create_model(activation='relu'):
        model = Sequential()
        model.add(Embedding(vocab_size, 16))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(16, kernel_initializer=init_mode, activation=activation))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = RMSprop(lr=learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(activation=activation)
    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
    # Best: 0.880533 using {'activation': 'softplus'}

activation='softplus'

if a:
    def create_model(neurons=1):
        model = Sequential()
        model.add(Embedding(vocab_size, 16))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(neurons, kernel_initializer=init_mode, activation=activation))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = RMSprop(lr=learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    neurons = [8, 10, 16, 32, 64, 128, 256, 512]
    neurons = [30, 31, 32, 33, 34, 35, 36]
    neurons = [32]
    param_grid = dict(neurons=neurons)
    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
#Best: 0.881667 using {'neurons': 32}

neurons = 32


from keras.constraints import maxnorm

if a:
    def create_model(weight_constraint=1):
        model = Sequential()
        model.add(Embedding(vocab_size, 16))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(neurons, kernel_initializer=init_mode, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
        optimizer = RMSprop(lr=learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    weight_constraint = [1, 2, 3, 4, 5]
    param_grid = dict( weight_constraint=weight_constraint)
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
#Best: 0.787533 using {'dropout_rate': 0.0, 'weight_constraint': 4}

weight_constraint=4


grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1) 
grid_result = grid.fit (partial_x_train, partial_y_train) 
# summarize results 
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) 
print()

