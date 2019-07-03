"""

Author: Ruixian Zhao

 This Helper Three script contains the functions to analyses the model
 It contains functions to get the accuracy and also output the figure about difference during training.

 More specifically:
 get_acc, save_pic, draw_pic, analyse_model

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# prepare the dataset
from helper1 import get_dataset, prepare_imdb

dataset = get_dataset('imdb')

(train_data, train_labels), (test_data, test_labels) = prepare_imdb(dataset)


# default it only returns the accuracy of the model
def get_acc(model, epochs=40, batch_size=512,
            complier=False, fit=False):
    model = model

    if complier:
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    if fit:
        model.fit(partial_x_train, partial_y_train, epochs=epochs,
                  batch_size=batch_size, validation_data=(x_val, y_val),
                  verbose=1)

    loss, acc = model.evaluate(test_data, test_labels)
    print("Accuracy: {:5.2f}%, Loos: {}".format(100 * acc, loss))


# functions to save the picture in high quality
def save_pic(title):
    path = title + ".png"
    print("Saving figure", title)
    plt.savefig(path, transparent=True, dpi=1200)


# functions to draw the figure about the difference between training and validation
def draw_pic(model, history, which, save=False, epochs='', batch_size='', other=''):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_x = range(1, len(acc) + 1)

    if which == 'Loss':
        plt.plot(epochs_x, loss, 'bo', label='Training loss')
        plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
        title = 'Training and Validation Loss\n'
        plt.title(title+'Epochs='+epochs+' Batch size='+batch_size+other)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if save:
            save_pic(title+'Epochs='+epochs+' Batch size='+batch_size+other)
        plt.show()
    elif which == 'Accuracy':
        plt.plot(epochs_x, acc, 'bo', label='Training acc')
        plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
        title = 'Training and Validation Accuracy\n'
        plt.title(title+'Epochs='+epochs+' Batch size='+batch_size+other)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if save:
            save_pic(title+'Epochs='+epochs+' Batch size='+batch_size+other)
        plt.show()
    elif which == 'both':
        plt.plot(epochs_x, loss, 'bo', label='Training loss')
        plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
        title = 'Training and Validation Loss\n'
        plt.title(title+'Epochs='+epochs+' Batch size='+batch_size+other)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if save:
            save_pic(title+'Epochs='+epochs+' Batch size='+batch_size+other)
        plt.show()

        plt.plot(epochs_x, acc, 'bo', label='Training acc')
        plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
        title = 'Training and Validation Accuracy\n'
        plt.title(title+'Epochs='+epochs+' Batch size='+batch_size+other)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if save:
            save_pic(title+'Epochs='+epochs+' Batch size='+batch_size+other)
        plt.show()
    else:
        print('Didn\'t draw any picture')

from keras import optimizers
# analyses the model and can output the figures
def analyse_model(model, epochs=40, batch_size=512, lr = 0.001, which='both',
                  complier=True, fit=True, draw=False, save=False, other = ''):
    model = model

    if complier:
        adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    if fit:
        start_time = time.time()
        history = model.fit(partial_x_train, partial_y_train, epochs=epochs,
                            batch_size=batch_size, validation_data=(x_val, y_val),
                            verbose=1)
        end_time = time.time() - start_time
        print("The whole time in trainning is: {:5.2f} seconds".format(end_time))
    if draw:
        if fit:
            draw_pic(model, history, which=which, epochs=str(epochs), batch_size=str(batch_size), save=save, other=other)
        else:
            print('If need to get the Training and validation loss figure, fitting in this function is required')

    loss, acc = model.evaluate(test_data, test_labels)
    print("Test Accuracy: {:5.2f}%".format(100 * acc))
    return acc*100


# test this script
if __name__ == '__main__':

    from tensorflow import keras
    from helper1 import get_dataset, prepare_imdb

    # prepare the dataset
    dataset = get_dataset('imdb')

    (train_data, train_labels), (test_data, test_labels), (x_val, partial_x_train), (y_val, partial_y_train) = prepare_imdb(dataset)

    # build a simple model
    vocab_size = 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.summary()

    # using the function to analyses the model
    analyse_model(model)
    # Accuracy: 87.68%, Loos: 0.3345
