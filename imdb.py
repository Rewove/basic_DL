"""

Author: Ruixian Zhao

 This imdb.py script can run and test all the models mentioned in report.

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
   5              imdb.py 4 0.002 14 512 7
   6              imdb.py 5 0.002 10 500 7
   7              imdb.py 4 0.002 16 512 7
   8              imdb.py 5 0.005 10 500 7




 The output of this script will be:

 1. The log of tested model and saved into folder logs, if there not have such folder it will create one.
 2. Save the whole model into ckpt format in current work directory.
 3. Print out the training accuracy, test accuracy of this model, compare with the records in the report.
 4. Output the training accuracy, test accuracy into results.txt, and also the records in the report.

"""
import argparse
import numpy as np

from helper2 import *  # all the models

def check_param_is_numeric(param, value):
    '''
    This function checks the input parameter whether numeric
    :param param: the input parameter get from argparse
    :param value: transfer the parameters into floating so that can be used in programs
    :return: the floating format of the input
    '''
    try:
        value = float(value)
    except:
        print("{} must be numeric".format(param))
        quit(1)
    return value


def main(combination = 1, learning_rate = 0.001, epochs = 24, batches = 100, seed = 7):
    np.random.seed(seed)
    print("Seed: {}".format(seed))

    if combination == 1:

        model_one(lr = learning_rate, epochs = epochs, batchs = batches)

    elif combination == 2:

        model_two(lr = learning_rate, epochs = epochs, batchs = batches)

    elif combination == 3:

        if epochs == 20:

            model_three(lr = learning_rate, epochs = epochs, batchs = batches)

        else:

            model_four(lr = learning_rate, epochs = epochs, batchs = batches)

    elif combination == 4:

        if epochs == 16:

            model_seven(lr = learning_rate, epochs = epochs, batchs = batches)

        else:

            model_five(lr = learning_rate, epochs = epochs, batchs = batches)

    elif combination == 5:

        if learning_rate == 0.002:

            model_six(lr = learning_rate, epochs = epochs, batchs = batches)

        else:

            model_eight(lr = learning_rate, epochs = epochs, batchs = batches)

    else:

        print("There must be some wrong in the combination parameters, please read the instructions/reports.")




if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("iterations", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")

    args = arg_parser.parse_args()

    combination = check_param_is_numeric("combination", args.combination)
    learning_rate = check_param_is_numeric("learning_rate", args.learning_rate)
    epochs = check_param_is_numeric("epochs", args.iterations)
    batches = check_param_is_numeric("batches", args.batches)
    seed = check_param_is_numeric("seed", args.seed)

    main(combination, learning_rate, int(epochs), int(batches), int(seed))
