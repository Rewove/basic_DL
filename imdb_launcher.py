"""

Author: Ruixian Zhao

 This IMDB launcher script will launch the best model for IMDB Review dataset.
 And output the test accuracy into a file.

"""

from tensorflow import keras
from helper3 import analyse_model

# data set IMDB
model_name = './imdb-final.ckpt'
new_model = keras.models.load_model(model_name)
acc = analyse_model(new_model, complier=False, fit=False)

print('The best model\'s Test Accuracy in the report is (89%) 88.54%')

with open('./results.txt', 'a') as f:
    print('Saving the results to file resutls.txt ...')
    f.write('=======================================================================\n')
    f.write('This is the results for reloading the model of IMDB dataset: \n')
    f.write('The best model\'s Test Accuracy in the report is (89%) 88.54%\n')
    f.write('Reload this model it get Test accuracy: {:5.2f}%\n'.format(acc))
    f.write('=======================================================================\n')
    print('The results has been save to file resutls.txt!')
