"""

Author: Ruixian Zhao

 This fashion launcher script will launch the best model for fashion MNIST dataset.
 And output the test accuracy into a file.

"""
from tensorflow import keras
from helper6 import prepare_fashion



# data set Fashion MNIST
(x_train, y_train), (test_data, test_labels) = prepare_fashion()

model_name = './fashion-final.ckpt'
new_model = keras.models.load_model(model_name)
score = new_model.evaluate(test_data, test_labels)

print("Test Accuracy: {:5.2f}%".format(100 * score[1]))

print('The best model\'s Test Accuracy in the report is 93% (92.6%)')

with open('./results.txt', 'a') as f:
    print('Saving the results to file resutls.txt ...')
    f.write('=======================================================================\n')
    f.write('This is the results for reloading the model of IMDB dataset: \n')
    f.write('The best model\'s Test Accuracy in the report is 93% (92.6%)\n')
    f.write('Reload this model it get Test accuracy: {:5.2f}%\n'.format(100 * score[0]))
    f.write('=======================================================================\n')
    print('The results has been save to file resutls.txt!')
