"""

Author: Ruixian Zhao

 This Helper Two script contains the functions to build and compile the models for IMDB
 It contains . More specifically:

 save_tensorboard: save the logs into tensorboard
 training: training the model in given epoch and batchs


 For the models in report, the parameters:

 Model ID      Combination      Learning-rate        Epochs       Batch-size      Seed      Notes
    1               1               0.001              24            100            7
    2               2               0.002              5             128            7
    3               3               0.002              20            64             7
    4               3               0.002              19            500            7
    5               4               0.002              14            512            7
    6               5               0.002              10            500            7
    7               4               0.002              16            512            7
    8               5               0.005              10            500            7

 To run the models, the commands is:

 Model ID                 Commands
   1 (best)       imdb.py 1 0.001 24 100 7
   2              imdb.py 2 0.002 5 128 7
   3              imdb.py 3 0.002 20 64 7
   4              imdb.py 3 0.002 19 500 7
   5              imdb.py 3 0.002 14 512 7
   6              imdb.py 5 0.002 10 500 7
   7              imdb.py 4 0.002 16 512 7
   8              imdb.py 5 0.005 10 500 7



 The output of this script will be:

 1. The log of tested model and saved into folder logs, if there not have such folder it will create one.
 2. Save the whole model into ckpt format in current work directory.
 3. Print out the training accuracy, test accuracy of this model, compare with the records in the report.
 4. Output the training accuracy, test accuracy into results.txt, and also the records in the report.


"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard

from keras.models import Sequential
from keras.layers import  GlobalAveragePooling1D, LSTM, GRU, AveragePooling1D
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras import regularizers
from keras.optimizers import RMSprop, Adam
from keras.optimizers import Adamax
from keras.constraints import maxnorm
from keras.callbacks import TensorBoard

from helper1 import *
from helper3 import *
from helper5 import *

import time

dataset = 'imdb'
vocab_size = 10000
max_words = 256
main_path = './'

# save model logs to tensorborad
log_filepath = 'logs'
def save_tensorboard (log_filepath, batch_size):
    tensorboard = TensorBoard(log_dir=log_filepath, histogram_freq=0,
                              batch_size=batch_size, write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    return tensorboard


# training the model and record the time
def training(model, batch_size, epochs, tensorboard):
    start = time.time()
    history = model.fit(train_data, train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(test_data, test_labels),
                        callbacks=[tensorboard])
    end = time.time()
    training_time = end - start
    return history, training_time


# get the test accuracy and training accuracy
def get_accuracy(model, history):
    score = model.evaluate(test_data, test_labels, verbose=0)
    trainaccuracy = history.history['acc'][-1]
    testaccuracy = score[1]
    return trainaccuracy, testaccuracy


# print out the accuracy and compare with report
def print_out(train_accuracy, test_accuracy, report_train,report_test):
    print('Training Accuracy: {:5.2f}%'.format(100 * train_accuracy))
    print('Test accuracy: {:5.2f}%'.format(100 * test_accuracy))
    print('In the report: Training Accuracy: {:5.2f}%'.format(report_train))
    print('In the report: Test accuracy: {:5.2f}%'.format(report_test))


##################################################################################
#################################BUILD MODEL HERE#################################
##################################################################################

# model 1:
def model_one(lr, epochs, batchs):
    print("The model one: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))

    # basic parameters
    comb = 1
    notation = 'Optimizer: RMSprop,\n Init_model: he_nromal,\n Activation: softplus'
    epochs = epochs
    batch_size = batchs
    lr = lr

    # self only parameters
    num = 'm1'
    log_filepath = 'logs'
    init_mode = 'he_normal'
    activation = 'softplus'

    # report records
    report_name = 'imdb-1-0.001-14-100-7-m1.ckpt'
    report_train = 93.98
    report_test = 88.54



    # build the model
    model = Sequential()
    model.add(Embedding(vocab_size, 16))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))



    # compile the model
    optimizer = RMSprop(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])



    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, batch_size, epochs, tensorboard)

    # evaluate the model
    train_accuracy, test_accuracy = get_accuracy(model, history)
    print_out(train_accuracy, test_accuracy, report_train, report_test)

    # save the model into main path
    model_saving, name_param = saving_model(model, dataset=dataset, comb=comb, epochs=epochs, batch_size=batch_size,
                                            lr=lr, seed=7, main_path=main_path, other=num)

    # output the results into file results.txt for manual marking
    output_results(model, model_saving=model_saving, name_param=name_param, time_=training_time,
                   train=train_accuracy, test=test_accuracy, report_name=report_name,
                   report_train=report_train, report_test=report_test, main_path=main_path,
                   seperate=False, save=True, dataset=dataset, notation=notation)


def model_two(lr, epochs, batchs):
    print("The model two: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))

    # basic parameters
    comb = 2
    notation = 'optimizer: RMSprop,\n init_model: he_nromal,\n activatin: softplus'
    epochs = epochs
    batch_size = batchs
    lr = lr

    # self only parameters
    num = 'm2'
    log_filepath = 'logs'
    init_mode = 'lecun_uniform'
    activation = 'softsign'

    # report records
    report_name = 'imdb-2-0.002-5-128-7-m2.ckpt'
    report_train = 95.98
    report_test = 87.23

    # build the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_words))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(20, kernel_initializer=init_mode, activation=activation,
                    kernel_constraint=maxnorm(2)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))

    # compile the model
    optimizer = Adamax(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, batch_size, epochs, tensorboard)

    # evaluate the model
    train_accuracy, test_accuracy = get_accuracy(model, history)
    print_out(train_accuracy, test_accuracy, report_train, report_test)

    # save the model into main path
    model_saving, name_param = saving_model(model, dataset=dataset, comb=comb, epochs=epochs, batch_size=batch_size,
                                            lr=lr, seed=7, main_path=main_path, other=num)

    # output the results into file results.txt for manual marking
    output_results(model, model_saving=model_saving, name_param=name_param, time_=training_time,
                   train=train_accuracy, test=test_accuracy, report_name=report_name,
                   report_train=report_train, report_test=report_test, main_path=main_path,
                   seperate=False, save=True, dataset=dataset, notation=notation)


def model_three(lr, epochs, batchs):
    print("The model three: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    # basic parameters
    comb = 3
    notation = 'optimizer: Adam,\n dr=0.2'
    epochs = epochs
    batch_size = batchs
    lr = lr

    # self only parameters
    num = 'm3'
    log_filepath = 'logs'


    # report records
    report_name = 'imdb-3-0.002-9-64-7-m3.ckpt'
    report_train = 95.42
    report_test = 85.42

    # build the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_words))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, batch_size, epochs, tensorboard)

    # evaluate the model
    train_accuracy, test_accuracy = get_accuracy(model, history)
    print_out(train_accuracy, test_accuracy, report_train, report_test)

    # save the model into main path
    model_saving, name_param = saving_model(model, dataset=dataset, comb=comb, epochs=epochs, batch_size=batch_size,
                                            lr=lr, seed=7, main_path=main_path, other=num)

    # output the results into file results.txt for manual marking
    output_results(model, model_saving=model_saving, name_param=name_param, time_=training_time,
                   train=train_accuracy, test=test_accuracy, report_name=report_name,
                   report_train=report_train, report_test=report_test, main_path=main_path,
                   seperate=False, save=True, dataset=dataset, notation=notation)


def model_four(lr, epochs, batchs):
    print("The model four: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    # basic parameters
    comb = 3
    notation = 'optimizer: Adam, dr=0.2 \n This models\' results various everytime!!!!'
    epochs = epochs
    batch_size = batchs
    lr = lr

    # self only parameters
    num = 'm4'
    log_filepath = 'logs'

    # report records
    report_name = 'imdb-3-0.002-19-500-7-m4.ckpt'
    report_train = 82.71
    report_test = 78.84

    # build the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_words))
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))


    # compile the model
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, batch_size, epochs, tensorboard)

    # evaluate the model
    train_accuracy, test_accuracy = get_accuracy(model, history)
    print_out(train_accuracy, test_accuracy, report_train, report_test)

    # save the model into main path
    model_saving, name_param = saving_model(model, dataset=dataset, comb=comb, epochs=epochs, batch_size=batch_size,
                                            lr=lr, seed=7, main_path=main_path, other=num)

    # output the results into file results.txt for manual marking
    output_results(model, model_saving=model_saving, name_param=name_param, time_=training_time,
                   train=train_accuracy, test=test_accuracy, report_name=report_name,
                   report_train=report_train, report_test=report_test, main_path=main_path,
                   seperate=False, save=True, dataset=dataset, notation=notation)


def model_five(lr, epochs, batchs):
    print("The model five: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    # basic parameters
    comb = 4
    notation = 'Optimizer: Adam'
    epochs = epochs
    batch_size = batchs
    lr = lr

    # self only parameters
    num = 'm5'
    log_filepath = 'logs'


    # report records
    report_name = 'imdb-4-0.002-14-512-7-m5.ckpt'
    report_train = 100
    report_test = 86.18

    # build the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, batch_size, epochs, tensorboard)

    # evaluate the model
    train_accuracy, test_accuracy = get_accuracy(model, history)
    print_out(train_accuracy, test_accuracy, report_train, report_test)

    # save the model into main path
    model_saving, name_param = saving_model(model, dataset=dataset, comb=comb, epochs=epochs, batch_size=batch_size,
                                            lr=lr, seed=7, main_path=main_path, other=num)

    # output the results into file results.txt for manual marking
    output_results(model, model_saving=model_saving, name_param=name_param, time_=training_time,
                   train=train_accuracy, test=test_accuracy, report_name=report_name,
                   report_train=report_train, report_test=report_test, main_path=main_path,
                   seperate=False, save=True, dataset=dataset, notation=notation)


def model_six(lr, epochs, batchs):
    print("The model six: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    # basic parameters
    comb = 5
    notation = 'Optimizer: Adam'
    epochs = epochs
    batch_size = batchs
    lr = lr

    # self only parameters
    num = 'm6'
    log_filepath = 'logs'

    # report records
    report_name = 'imdb-5-0.002-10-500-7-m6.ckpt'
    report_train = 99.20
    report_test = 85.6

    # build the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_words))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, batch_size, epochs, tensorboard)

    # evaluate the model
    train_accuracy, test_accuracy = get_accuracy(model, history)
    print_out(train_accuracy, test_accuracy, report_train, report_test)

    # save the model into main path
    model_saving, name_param = saving_model(model, dataset=dataset, comb=comb, epochs=epochs, batch_size=batch_size,
                                            lr=lr, seed=7, main_path=main_path, other=num)

    # output the results into file results.txt for manual marking
    output_results(model, model_saving=model_saving, name_param=name_param, time_=training_time,
                   train=train_accuracy, test=test_accuracy, report_name=report_name,
                   report_train=report_train, report_test=report_test, main_path=main_path,
                   seperate=False, save=True, dataset=dataset, notation=notation)


def model_seven(lr, epochs, batchs):
    print("The model seven: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    # basic parameters
    comb = 4
    notation = 'Optimizer: Adam, \n Drop rate: 0.5'
    epochs = epochs
    batch_size = batchs
    lr = lr

    # self only parameters
    num = 'm7'
    log_filepath = 'logs'

    # report records
    report_name = 'imdb-4-0.002-5-512-7-m7.ckpt'
    report_train = 100
    report_test = 86.18

    # build the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, batch_size, epochs, tensorboard)

    # evaluate the model
    train_accuracy, test_accuracy = get_accuracy(model, history)
    print_out(train_accuracy, test_accuracy, report_train, report_test)

    # save the model into main path
    model_saving, name_param = saving_model(model, dataset=dataset, comb=comb, epochs=epochs, batch_size=batch_size,
                                            lr=lr, seed=7, main_path=main_path, other=num)

    # output the results into file results.txt for manual marking
    output_results(model, model_saving=model_saving, name_param=name_param, time_=training_time,
                   train=train_accuracy, test=test_accuracy, report_name=report_name,
                   report_train=report_train, report_test=report_test, main_path=main_path,
                   seperate=False, save=True, dataset=dataset, notation=notation)


def model_eight(lr, epochs, batchs):
    print("The model eight: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    # basic parameters
    comb = 5
    notation = 'Optimizer: Adam,\n Combine all the layers'
    epochs = epochs
    batch_size = batchs
    lr = lr

    # self only parameters
    num = 'm8'
    log_filepath = 'logs'

    # report records
    report_name = 'imdb-5-0.005-10-500-7-m8.ckpt'
    report_train = 96.5
    report_test = 86.9

    # build the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_words))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(AveragePooling1D())
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.05)))  # relu
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, batch_size, epochs, tensorboard)

    # evaluate the model
    train_accuracy, test_accuracy = get_accuracy(model, history)
    print_out(train_accuracy, test_accuracy, report_train, report_test)

    # save the model into main path
    model_saving, name_param = saving_model(model, dataset=dataset, comb=comb, epochs=epochs, batch_size=batch_size,
                                            lr=lr, seed=7, main_path=main_path, other=num)

    # output the results into file results.txt for manual marking
    output_results(model, model_saving=model_saving, name_param=name_param, time_=training_time,
                   train=train_accuracy, test=test_accuracy, report_name=report_name,
                   report_train=report_train, report_test=report_test, main_path=main_path,
                   seperate=False, save=True, dataset=dataset, notation=notation)


