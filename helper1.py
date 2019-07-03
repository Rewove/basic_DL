"""

 This Helper One script contains the functions to prepare the [IMDB] data
 It contains data download, split data set and data decode. More specifically:

 get_dataset, prepare_imdb, prepare_word_dic, index_word, decode_review

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

'''
 [DOWNLOADER] & [SLICER]
 download the data and prepare it to be used
 return train_data, train_labels, test_data and test_labels
'''


# downloader
def get_dataset(name):
    '''
    This function download the dataset.
    :param name: determine which dataset to be downloaded.
    :return: the dataset
    '''
    from tensorflow import keras
    if name == 'imdb'or'IMDB':
        dataset = keras.datasets.imdb
    elif name == 'fashion'or'Fashion':
        dataset = keras.datasets.fashion_mnist
    else:
        print('There is no dataset named {} has been downloaded \n.'.format(name))
    print("The {} dataset has been downloaded successfully. \n".format(name))
    return dataset


# slicer
def prepare_imdb(dataset, cross_val=False, partial=False):
    '''
    This function will prepare the dataset for training and testing.
    If cross validation is True then it will divided the training set into validate set and partial set.
    This is used to cross valid the model while not using the testing set.
    It is default to be False.
    :param dataset: the dataset to be prepared.
    :param cross_val: decide whether to divided the training set.
    :param partial: decided whether output all the dataset or only part or them.
    :return:
    '''
    data = dataset
    # split the dataset and drop the rare words
    (train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
    print("The dataset has been split into training-set and testing-set successfully. \n")

    # transfer into same length
    word_index = data.get_word_index()
    word_index = prepare_word_dic(word_index)

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    train_data = pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
    test_data = pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

    if cross_val:
        # creat testing set, instead of using the testing set
        x_val = train_data[:10000]
        partial_x_train = train_data[10000:]

        y_val = train_labels[:10000]
        partial_y_train = train_labels[10000:]
        print("The testing set has been created from training-set. \n")

        if partial:
            return (x_val, partial_x_train), (y_val, partial_y_train)
        else:
            return (train_data, train_labels), (test_data, test_labels), (x_val, partial_x_train), (y_val, partial_y_train)
    else:
        return (train_data, train_labels), (test_data, test_labels)




'''
 [WORD INDEX]
 get the reverse word index
 and prepare the first 3 reserved keys
'''


# filling the first three reservation key
def prepare_word_dic(word_index):
    '''
    This function will handle with the first reservation keys, otherwise directly translate will looks strange
    :param word_index: the word_index dictionary of the dataset
    :return: filling the first few reservation keys with PAD, START, UNK and UNUSED.
    '''
    word_index = {key: (value + 3) for key, value in word_index.items()}
    word_index["<PAD>"] = 0  # filling
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3  # unused
    return word_index


# get the word via its key
def index_word(dataset):
    '''
    This function is to revise the word_index in order to use it to translate the integer data.
    It first need to get the word_index dictionary
    and then handle with the first reservation keys
    Finally it will revise the word_index into index_word
    :param dataset: the dataset used here
    :return: return the reversed word index dictionary
    '''
    data = dataset
    word_index = data.get_word_index()

    word_index = prepare_word_dic(word_index)

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return reverse_word_index


'''
 [DECODER]
 decode the data into sentence
'''


def decode_review(text, index_word, show=False):
    '''
    This function will decode the integer data into readable sentence.
    :param text: the integer data need to be decoded (translate)
    :param index_word: the index to word dictionary we get above
    :param show: decided whether to print out the translated sentence
    :return: the readable sentence
    '''
    index_word = index_word
    sentence = ' '
    sentence = sentence.join([index_word.get(i, '?') for i in text])
    if show:
        print(sentence)
    return sentence


'''
 test this script
'''

if __name__ == '__main__':

    dataset = get_dataset('imdb')

    (train_data, train_labels), (test_data, test_labels) = prepare_imdb(dataset)

    reverse_word_index = index_word(dataset)

    # show if things went well
    print('The first instance in the dataset is:')
    print(train_data[0])
    print()
    print('Decode the first instance into same-length sentence:')
    decode_review(train_data[0], reverse_word_index, True)

