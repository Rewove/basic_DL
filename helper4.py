"""

Author: Ruixian Zhao

 This Helper Four script contains the functions to build the model for fashion Dataset.
 It contains . More specifically:

 save_tensorboard: save the logs into tensorboard
 training: training the model in given epoch and batchs


 For the models in report, the parameters:

 Model ID      Combination      Learning-rate        Epochs       Batch-size      Seed      Notes
    1               1 (4)           0.001              18            256            7         -     (not run)
    2               1               0.001              24            128            7         *     (run)
    3               1 (4)           0.001              24            128            7         *     (not run)
    4               1               0.001              16            128            7
    5               1               0.001              15            256            7
    6               1               0.001              18            256            7         -     (run)
    7               2               0.001              15            64             7
    8               2               0.001              10            256            7
    9               3               0.001              16            64             7
    10              3               0.001              30            32             7

 Only see from these parameters the model 1 is same with model 6, and model 2, 3 also. So if you using
 these parameters to run the model, that is the model 2 and 6 will run. If you want to run 1 and 3, you
 need to set the combination as 4. i.e. "fashion.py 4 0.002 18 256 7" for model 1.
 Sorry for this inconvenience.

 To run the models, the commands is:

 Model ID                   Commands
   1              fashion.py 4 0.001 18 256 7
   2              fashion.py 1 0.001 24 128 7
   3              fashion.py 4 0.001 24 128 7
   4 (best)       fashion.py 1 0.001 16 128 7 (default)
   5              fashion.py 1 0.001 15 256 7
   6              fashion.py 1 0.001 18 256 7
   7              fashion.py 2 0.001 15 64 7
   8              fashion.py 2 0.001 10 256 7
   9              fashion.py 3 0.001 16 64 7
   10             fashion.py 3 0.001 30 32 7



 The output of this script will be:

 1. The log of tested model and saved into folder logs, if there not have such folder it will create one.
 2. Save the whole model into ckpt format in current work directory.
 3. Print out the training accuracy, test accuracy of this model, compare with the records in the report.
 4. Output the training accuracy, test accuracy into results.txt, and also the records in the report.



"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, InputLayer, MaxPool2D
from keras.layers import Dropout, MaxPooling2D, AveragePooling2D
from keras.initializers import Constant
from keras.optimizers import Adam
import time

dataset='fashion'
main_path = './'
log_filepath = 'logs'
num_classes =10


from helper5 import *
from helper6 import prepare_fashion
(x_train, y_train), (x_test, y_test) = prepare_fashion()

from keras.callbacks import TensorBoard
def save_tensorboard (log_filepath, batch_size):
    tensorboard = TensorBoard(log_dir=log_filepath, histogram_freq=0,
                              batch_size=batch_size, write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    return tensorboard

# compile and training the model
def training(model, lr, epochs, batch_size, tensorboard):
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    start = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[tensorboard])
    end = time.time()

    training_time = end - start

    return history, training_time


# get the test accuracy and training accuracy
def get_accuracy(model, history):
    score = model.evaluate(x_test, y_test, verbose=0)
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


def model_ten(lr, epochs, batchs):
    print("The model one: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    epochs = epochs
    batch_size = batchs
    log_filepath = 'logs'

    # self notification
    notation = '32, 64; 16, 32 in conv1 and conv2; 64 double Dense with dr 0.5'
    comb=4
    num = 'm1'

    # report records
    report_name = 'fashion-1-0.001-16-256-7-m1.ckpt'
    report_train = 89.08
    report_test = 91.21


    # build the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, lr, epochs, batch_size, tensorboard)

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
    print("The model two: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    epochs = epochs
    batch_size = batchs
    log_filepath = 'logs'

    # self notification
    notation = '64; 64, 64 in conv1 and conv2; 256 double Dense with dr 0.4'
    comb = 1
    num = 'm2'

    # report records
    report_name = 'fashion-1-0.001-24-128-7-m2.ckpt'
    report_train = 97.69
    report_test = 92.34

    # build the model
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))



    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, lr, epochs, batch_size, tensorboard)

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


def model_nine(lr, epochs, batchs):
    print("The model three: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    epochs = epochs
    batch_size = batchs
    log_filepath = 'logs'

    # self notification
    notation = '64, 64; 32, 32 in conv1 and conv2; 128 double Dense with dr 0.4'
    comb = 4
    num = 'm3'

    # report records
    report_name = 'fashion-1-0.001-24-128-7-m3.ckpt'
    report_train = 95.86
    report_test = 92.62

    # build the model
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=2, bias_initializer=Constant(0.01), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=2, bias_initializer=Constant(0.01), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, lr, epochs, batch_size, tensorboard)

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
    epochs = epochs
    batch_size = batchs
    log_filepath = 'logs'

    # self notification
    notation = '32, 32; 32, 32 in conv1 and conv2; 100 double Dense with dr 0.4'
    comb = 1
    num = 'm4'

    # report records
    report_name = 'fashion-1-0.001-16-128-7-m4.ckpt'
    report_train = 93.81
    report_test = 92.64

    # build the model
    model = Sequential()
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, lr, epochs, batch_size, tensorboard)

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


def model_one(lr, epochs, batchs):
    print("The model five: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    epochs = epochs
    batch_size = batchs
    log_filepath = 'logs'
    # self notification
    notation = '32, 64; 32 in conv1 and conv2; 100 double Dense with dr 0.4'
    comb = 1
    num = 'm5'

    # report records
    report_name = 'fashion-1-0.001-15-256-7-m5.ckpt'
    report_train = 93.17
    report_test = 92.10

    # build the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, lr, epochs, batch_size, tensorboard)

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
    print("The model six: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    epochs = epochs
    batch_size = batchs
    log_filepath = 'logs'

    # self notification
    notation = '32; 32 in conv1 and conv2; 100 double Dense with dr 0.4'
    comb = 1
    num = 'm6'

    # report records
    report_name = 'fashion-1-0.001-18-128-7-m6.ckpt'
    report_train = 96.31
    report_test = 91.67

    # build the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, lr, epochs, batch_size, tensorboard)

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
    print("The model seven: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    epochs = epochs
    batch_size = batchs
    log_filepath = 'logs'

    # self notification
    notation = '32 32 in conv with constant bias; 128 single Dense'
    comb = 2
    num = 'm7'

    # report records
    report_name = 'fashion-2-0.001-15-64-7-m7.ckpt'
    report_train = 98.81
    report_test = 90.95

    # build the model
    model = Sequential()
    model.add(InputLayer(input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (2, 2), padding='same', bias_initializer=Constant(0.01), kernel_initializer='random_uniform'))
    model.add(Conv2D(32, (2, 2), padding='same', bias_initializer=Constant(0.01), kernel_initializer='random_uniform',
                     input_shape=(28, 28, 1)))
    model.add(MaxPool2D(padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', bias_initializer=Constant(0.01), kernel_initializer='random_uniform'))
    model.add(Dense(10, activation='softmax'))

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, lr, epochs, batch_size, tensorboard)

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
    print("The model eight: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    epochs = epochs
    batch_size = batchs
    log_filepath = 'logs'
    # self notification
    notation = 'single 64 in conv with Average Pooling; 64 double Dense'
    comb = 2
    num = 'm8'

    # report records
    report_name = 'fashion-2-0.001-10-256-7-m8.ckpt'
    report_train = 92.17
    report_test = 90.05

    # build the model
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(AveragePooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, lr, epochs, batch_size, tensorboard)

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
    print("The model nine: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    epochs = epochs
    batch_size = batchs
    log_filepath = 'logs'
    # self notification
    notation = '256 double Dense with 0.5 drop out rate'
    comb = 3
    num = 'm9'

    # report records
    report_name = 'fashion-3-0.001-16-64-7-m9.ckpt'
    report_train = 87.37
    report_test = 87.45

    # build the model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, lr, epochs, batch_size, tensorboard)

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
    print("The model ten: learning rate: {}, epochs: {}, batchs: {}".format(lr, epochs, batchs))
    epochs = epochs
    batch_size = batchs
    log_filepath = 'logs'
    # self notification
    notation = '256 double Dense with 0.5 drop out rate'
    comb = 3
    num = 'm10'

    # report records
    report_name = 'fashion-3-0.001-30-32-7-m10.ckpt'
    report_train = 87.70
    report_test = 88.30

    # build the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # prepare to save to tensorboard
    tensorboard = save_tensorboard(log_filepath, batch_size)

    # training the model and record the time
    history, training_time = training(model, lr, epochs, batch_size, tensorboard)

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
