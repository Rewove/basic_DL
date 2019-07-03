"""

Author: Ruixian Zhao

 This Helper Five script contains the functions to save the results.

 It can save the model in an appropriate way (with clearly marked names), and output the results
 to a txt file.
 More specifically:

 saving_model, output_results.

 In the txt file, there will be the parameters of the model, and the training accuracy test accuracy
 after training, and plus the results of model shows in the report.
 If you like you can also save the summary of the model to have a better view of its
 architecture.


"""
import os

def saving_model(model, dataset, comb, epochs, batch_size, lr, seed, main_path = './', other = ''):
    '''
    saving the model in an appropriate way.
    :param dataset: the dataset been used, at the begin of the name (imdb or fashion)
    :param comb:  the combinations of network
    :param epochs: the number of iterations to train the model
    :param batch_size: the batch size while training
    :param lr: the learning rate of training
    :param seed: the random seed in the purpose of reproduce
    :param other: some other notes you want to make distinguishable
    :return: model saved in name format: dataset-combination-lr-epochs-batch_size-seed-other.ckpt
             also return the model name for further purpose
    '''
    import os

    name_param = list([dataset, str(comb), str(lr), str(epochs), str(batch_size), str(seed), other])
    link = '-'
    model_name = link.join(name_param)
    fileformat = '.ckpt'
    model_saving = model_name + fileformat
    save_path = os.path.join(main_path, model_saving)

    if main_path == './':
        print('Saving model \"{}\" into currently work directory'.format(model_saving))
    else:
        try:
            os.makedirs(main_path)
            print('The folder {} has been created, now saving the model \"{}\"...'.format(main_path, model_saving))
        except FileExistsError:
            print('Saving model \"{}\" into folder {}.'.format(model_saving, main_path))

    model.save(save_path)
    return model_saving, name_param


def output_results(model, model_saving, name_param, time_ , train, test, report_name, report_train, report_test, main_path = './', seperate = False, save = False, dataset = '', notation = ''):
    '''
    output results to file "results.txt", or with marked by the dataset name ("dataset-results.txt")
    :param model: model used here to print out the summary of it, to get a fully understand of its architecture
    :param model_saving: the name of the model, get from function saving model
    :param name_param: the parameters of the model, get from function saving model
    :param time_: the training time of the model, in seconds, nee to input
    :param train: the training accuracy, will be transformed by multiple 100
    :param test:  the training accuracy, will be transformed by multiple 100
    :param report_name: the name of this model in the report
    :param report_train: the training accuracy of this model shows in the report
    :param report_test:  the test accuracy of this model shows in the report
    :param seperate: decide whether to save the file with the dataset name as the prefix
    :param save: decide whether to sae the model summary, default False because it will looks mess
    :param dataset: the name of the dataset used here
    :param notation: some other informations you want to add to the output file
    :return:
    '''
    import os
    results = 'results.txt'
    if seperate:
        link = '_'
        results = dataset+link+results
    result_file = os.path.join(main_path, results)
    # make sure the out put accuracy is the percentage
    if train < 2:
        train = train * 100
    if test < 2:
        test = test * 100
    if report_test < 2:
        test = report_test * 100
    if report_train < 2:
        test = report_train * 100
    with open(result_file, 'a') as f:
        print('Saving the parameters and results to file resutls.txt ...')
        f.write('=======================================================================\n')
        f.write('Model name: {}\n'.format(model_saving))
        # [dataset, str(comb), str(lr), str(epochs), str(batch_size), str(seed), other]
        f.write(' The dataset: {},\n The combination: {},\n Learning rate: {},\n Epochs: {},\n Batch size: {},\n Seed: {},\n Other notes: {}\n'.format(name_param[0],name_param[1],name_param[2],name_param[3],name_param[4],name_param[5],name_param[6]))
        f.write('Notations: {}\n'.format(notation))
        f.write('\n')
        f.write('Training time: {:5.2f} seconds\n'.format(time_))
        f.write('Training Accuracy: {:5.2f}%\n'.format(train))
        f.write('Test accuracy: {:5.2f}%\n'.format(test))
        f.write('\n')
        f.write('In the report the model name is: {}\n'.format(report_name))
        f.write('In the report the Training Accuracy: {:5.2f}%\n'.format(report_train))
        f.write('In the report the Test Accuracy: {:5.2f}%\n'.format(report_test))
        f.write('\n')
        if save:
            f.write('The Architecture of the Model:\n')
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write('=======================================================================\n')
        f.write('\n')
        print('All the information has been saved into results.txt! Go to check it!')
        print('It takes a while to finish everything, please wait a little bit while.')
