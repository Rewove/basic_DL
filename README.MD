Author: Ruixian Zhao

# Introduction
Team: Kappa
Team member: Ruixian Zhao
Contact Email: ruixian.zhao.2018@uni.strath.ac.uk

Dataset: IMDB Review, Fashion MNIST

This documentation will first introduce the files in this folder, and then give the instructions on how to run those scripts in an appropriate way, by giving explanations and examples.

# File Introduction
This folder includes the files and scripts used in this project:

1. **requirements.txt**: this includes all the packages with certain versions. Can be used to build virtual environments from it. This is highly recommended.
2. **fashion.py**: from this script you can test all the models mentioned in the report. Since the parameters used in this script is slightly different from the _example_, there will be an introduction later. By running this scrip, if running a specific model, the logs of this model will be saved into folder *logs*, and the training accuracy, testing accuracy will be out put to file *results.txt* with clearly marked. Also after training the whole model will be saved as ".ckpt" file in the work directory. The name fo the model will be the "dataset-combination-epochs-batch_size-learning_rate-seed_number_of_model.ckpt. If not run the script with a specific model, it will run the best model automatically.
3. **imdb.py**: similar with above but test all the models in imdb.
4. **imdb_launcher.py**: to launch the best model *imdb-final.ckpt* of IMDB Review dataset and out put the test accuracy to file *results.txt*.
5. **fashion_launcher.py**: to launch the best model *fashion-final.ckpt* of fashion dataset and out put the test accuracy to file *results.txt*.
6. **fashion-final.ckpt; imdb-final.ckpt**: as mentioned above these are the two best models of each dataset.
7. **folder: previous_logs**: This folder contains all the models logs which can use tensorboard to see details. The format of the name of the folder is "logs_dataset_number_of_mdoel". To use tensorborad to see the details, using ```tensorboard  --logdir="./previous_logs/logs_dataset_number_of_model" -- port 6006```. For example: ```tensorboard  --logdir="./previous_logs/logs_fashion_1" -- port 6006```.
8. **folder: fashion_models**: This folder contains all the models in this report for fashion dataset. The format of the name is: "dataset-combination-epochs-batch_size-learning_rate-seed_number_of_model.ckpt". To launch a model you can use code:

    ```python 
    from tensorflow import keras
    new_model = keras.models.load_model('./fashion_models/name-of-it.ckpt')
    ```
    Or just using the script named "launcher_fashion_models.py" to launch the model by its ID. For example ``` python ./fashion_models/launcher_fashion_models.py ID``` where the ID refers to the number of the models in the report.
9. **folder: imdb_models**: Similar with above but this is the folder for the IMDB dataset and there also have a script named "launcher_imdb_models.py" to help to lunch all the IMDB models mentioned in report.
10. **fashion_results.txt**:  This file includes all the information about training the models for fashion MNIST dataset. It contains the name fo the model, the notation for the model, the parameters setting for the model, the training time,  the training accuracy and test accuracy. Also the summary of the model has been included for further checking.
11. **imdb_results.txt**: Similar with illustration above but for the IMDB dataset.
12. **helper1.py**: This Helper One script contains the functions to prepare the [IMDB] data. It contains data download, split data set and data decode. More specifically: ```get_dataset, prepare_imdb, prepare_word_dic, index_word, decode_review```
13. **helper2.py**: This Helper Two script contains the functions to build the model for IMDB.
14. **helper3.py**: This Helper Three script contains the functions to analyses the model. It contains functions to get the accuracy and also output the figure about difference during training. More specifically: ```get_acc, save_pic, draw_pic, analyse_model```
15. **helper4.py**: This Helper Four script contains the functions to build the model for fashion Dataset.
16. **helper5.py**: This Helper Five script contains the functions to save the results.
17. **fun_classify.py**: This script is just for fun, by running this script it will launch the best model of IMDB. Then the program will use this model to help classify some simple sentences, to judge the mode of the sentence. The script will first print out the sentence collected from IMDB real reviews and print out the mode of this sentence the program this it is. It is really fun and this is really interesting to find that the model we build is working! Mostly it works very well.
18. **fun_input_classify.py**: This script is the advance version of above one, the program can take the input from user and judge the mode of the input sentence. It will first remove all the punctuations in the  input and then transfer them into lower case. After this preprocessing, the script will find the words in the word-index dictionary of IMDB dataset to transfer the input words into integers. Due to time limit the program can only process the sentence less than 200 words and cannot recognize some rare words. There probably be some problems the translation stage by using word-index dictionary. *Any way it is still very exciting to find that our model can recognize most of the modes correctly of what we input.*
19. **gridsearch1.py, gridsearch2.py**: This two scripts contains the raw code for grid search the best combinations of the parameters for the best two models of IMDB dataset.
20. **folder: raw_code**: This folder contains the raw code of this report. Please be caution to running them, there might have some issues by directly running them, only for the purpose of proofing that the whole work is been produced by those codes.
21. **_*(not created yet) folder: logs*_**: This folder will created by the script *fashion.py* or *imdb.py*, contains all the logs while training the model by running those scripts.
22. ***_(not created yet) results.txt_***: This file will be created by run the script and it will contains the training accuracy, testing accuracy with clearly marked.
23. **_*(not created yet) dataset-modelname.ckpt*_**: This files is the whole saved model after running the scripts. The format of its name has been talked above.

# imdb.py Instructions

This imdb.py script can run and test all the models mentioned in report.

 For the models in report, the parameters:
```
 Model ID      Combination      Learning-rate        Epochs       Batch-size      Seed      Notes
    1               1               0.001              24            100            7
    2               2               0.002              5             128            7
    3               3               0.002              20            64             7
    4               3               0.002              19            500            7
    5               4               0.002              14            512            7
    6               5               0.002              10            500            7
    7               4               0.002              16            512            7
    8               5               0.005              10            500            7
```
 To run the models, the commands is:
```
 Model ID                 Commands
   1 (best)       imdb.py 1 0.001 24 100 7
   2              imdb.py 2 0.002 5 128 7
   3              imdb.py 3 0.002 20 64 7
   4              imdb.py 3 0.002 19 500 7
   5              imdb.py 4 0.002 14 512 7
   6              imdb.py 5 0.002 10 500 7
   7              imdb.py 4 0.002 16 512 7
   8              imdb.py 5 0.005 10 500 7
```



 The output of this script will be:

 1. The log of tested model and saved into folder logs, if there not have such folder it will create one.
 2. Save the whole model into ckpt format in current work directory.
 3. Print out the training accuracy, test accuracy of this model, compare with the records in the report.
 4. Output the training accuracy, test accuracy into results.txt, and also the records in the report.

# imdb_launcher.py Instructions

 This IMDB launcher script will launch the best model for IMDB Review dataset.
 And output the test accuracy into a file.
 

# fashion.py Instructions


Author: Ruixian Zhao

 This fashion.py script can run and test all the models mentioned in report.

 For the models in report, the parameters:
```
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
```
 Only see from these parameters the model 1 is same with model 6, and model 2, 3 also. So if you using
 these parameters to run the model, that is the model 2 and 6 will run. If you want to run 1 and 3, you
 need to set the combination as 4. i.e. "fashion.py 4 0.002 18 256 7" for model 1.
 Sorry for this inconvenience.

 To run the models, the commands is:
```
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
```


 The output of this script will be:

 1. The log of tested model and saved into folder logs, if there not have such folder it will create one.
 2. Save the whole model into ckpt format in current work directory.
 3. Print out the training accuracy, test accuracy of this model, compare with the records in the report.
 4. Output the training accuracy, test accuracy into results.txt, and also the records in the report.

# fashion_launcher.py Instructions
 This fashion launcher script will launch the best model for fashion MNIST dataset.
 And output the test accuracy into a file.

# gridsearch1.py gridsearch2.py Instructions

This two scripts contains the raw code for grid search the best combinations of the parameters for the best two models of IMDB dataset.

# Check list
* [x] 1. **requirements.txt**
* [x] 2. **fashion.py**
* [x] 3. **imdb.py**
* [x] 4. **imdb_launcher.py**
* [x] 5. **fashion_launcher.py**
* [x] 6. **fashion-final.ckpt; imdb-final.ckpt**
* [x] 7. **folder: previous_logs**
* [x] 8. **folder: fashion_models**
* [x] 9. **folder: imdb_models**
* [x] 10. **fashion_results.txt**
* [x] 11. **imdb_results.txt**
* [x] 12. **helper1.py**
* [x] 13. **helper2.py**
* [x] 14. **helper3.py**
* [x] 15. **helper4.py**
* [x] 16. **helper5.py**
* [x] 17. **helper 6**
* [x] 18. **fun_classify.py**
* [x] 19. **fun_input_classify.py**
* [x] 20. **gridsearch1.py, gridsearch2.py**
* [x] 21. **folder: raw_code**
* [ ] 22. **_*(not created yet) folder: logs*_**
* [ ] 23. ***_(not created yet) results.txt_***
* [ ] 24. **_*(not created yet) dataset-modelname.ckpt*_**
