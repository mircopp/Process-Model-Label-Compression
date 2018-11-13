import json
import numpy as np

def save_preprocessed_data(data, chunksize=20, source='', name='preprocessed'):
    """
    Saves the preprocessed data
    :param data: The data to be saved
    :param chunksize: The chunksize of the data.
    :param source: The source folder.
    :param name: The data name.
    :return: None
    """
    for i in range(chunksize):
        with open(source + name + str(i) + '.json', 'w') as outfile:
            print('Saving: {}'.format(source + name + str(i) + '.json'))
            json.dump(data[i::chunksize].tolist(), outfile)
            outfile.close()


def load_preprocessed_data(chunksize = 20, source='', name = 'preprocessed'):
    """
    Loads the preprocessed data from disk
    :param chunksize: The chunksize of the data.
    :param source: The source folder
    :param name: The name of the files
    :return: The loaded data
    """
    result = np.array([])
    for i in range(chunksize):
        print('Reading: {}'.format(source + name + str(i) + '.json'))
        chunk = read_json(source + name + str(i) + '.json')
        if result.any():
            result = np.concatenate((result, chunk))
        else:
            result = chunk
    return result

def read_json(file_name):
    """
    Read a json file
    :param file_name: The name of the file.
    :return: The loaded json as dict.
    """
    datafile = open(file_name, 'r')
    data = datafile.read()
    datafile.close()
    del datafile
    return np.array(json.loads(data))

