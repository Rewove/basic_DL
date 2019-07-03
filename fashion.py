"""

Author: Ruixian Zhao

 This fashion.py script can run and test all the models mentioned in report.

 For the models in report, the parameters:

 Model ID      Combination      Learning-rate        Epochs       Batch-size      Seed      Notes
    1               1 (4)           0.002              18            256            7         -     (not run)
    2               1               0.002              24            128            7         *     (run)
    3               1 (4)           0.002              24            128            7         *     (not run)
    4               1               0.002              16            128            7
    5               1               0.002              15            256            7
    6               1               0.002              18            256            7         -     (run)
    7               2               0.002              15            64             7
    8               2               0.002              10            256            7
    9               3               0.002              16            64             7
    10              3               0.002              30            32             7

 Only see from these parameters the model 1 is same with model 6, and model 2, 3 also. So if you using
 these parameters to run the model, that is the model 2 and 6 will run. If you want to run 1 and 3, you
 need to set the combination as 4. i.e. "fashion.py 4 0.002 18 256 7" for model 1.
 Sorry for this inconvenience.

 To run the models, the commands is:

 Model ID                   Commands
   1              fashion.py 4 0.002 18 256 7
   2              fashion.py 1 0.002 24 128 7
   3              fashion.py 4 0.002 24 128 7
   4 (best)       fashion.py 1 0.002 16 128 7 (default)
   5              fashion.py 1 0.002 15 256 7
   6              fashion.py 1 0.002 18 256 7
   7              fashion.py 2 0.002 15 64 7
   8              fashion.py 2 0.002 10 256 7
   9              fashion.py 3 0.002 16 64 7
   10             fashion.py 3 0.002 30 32 7



 The output of this script will be:

 1. The log of tested model and saved into folder logs, if there not have such folder it will create one.
 2. Save the whole model into ckpt format in current work directory.
 3. Print out the training accuracy, test accuracy of this model, compare with the records in the report.
 4. Output the training accuracy, test accuracy into results.txt, and also the records in the report.

"""
import argparse
import numpy as np

from helper4 import *

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


def main(combination = 1, learning_rate = 0.002, epochs = 16, batches = 128, seed = 7):
    np.random.seed(seed)
    print("Seed: {}".format(seed))

    if combination == 1:

        if batches == 256:

            if epochs == 15:

                model_one(lr = learning_rate, epochs = epochs, batchs = batches)

            else:

                model_two(lr = learning_rate, epochs = epochs, batchs = batches)

        else:

            if epochs == 24:

                model_three(lr = learning_rate, epochs = epochs, batchs = batches)

            else:

                model_four(lr = learning_rate, epochs = epochs, batchs = batches)

    elif combination == 2:

        if epochs ==15:

            model_five(lr = learning_rate, epochs = epochs, batchs = batches)

        else:

            model_six(lr = learning_rate, epochs = epochs, batchs = batches)

    elif combination == 3:

        if epochs == 16:

            model_seven(lr = learning_rate, epochs = epochs, batchs = batches)

        else:

            model_eight(lr = learning_rate, epochs = epochs, batchs = batches)

    elif combination == 4:

        if epochs == 24:

            model_nine(lr = learning_rate, epochs = epochs, batchs = batches)

        else:

            model_ten(lr = learning_rate, epochs = epochs, batchs = batches)

    else:

        print("There must be some wrong in the inputted parameters, please read the instructions/reports.")




if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run", default='1')
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter", default='0.002')
    arg_parser.add_argument("iterations", help="Number of iterations to perform", default='16')
    arg_parser.add_argument("batches", help="Number of batches to use", default='128')
    arg_parser.add_argument("seed", help="Seed to initialize the network", default='7')

    args = arg_parser.parse_args()
    combination = check_param_is_numeric("combination", args.combination)
    learning_rate = check_param_is_numeric("learning_rate", args.learning_rate)
    epochs = check_param_is_numeric("epochs", args.iterations)
    batches = check_param_is_numeric("batches", args.batches)
    seed = check_param_is_numeric("seed", args.seed)


    main(combination, learning_rate, int(epochs), int(batches), int(seed))




