import json
import numpy as np
import sys

def save_preprocessed_data(data, chunksize=20, name='preprocessed'):
    for i in range(chunksize):
        with open('matrizes/' + name + str(i) + '.json', 'w') as outfile:
            print('Saving: {}'.format(name + str(i) + '.json'))
            json.dump(data[i::chunksize].tolist(), outfile)
            outfile.close()


def load_preprocessed_data(chunksize = 20, name = 'preprocessed'):
    result = np.array([])
    for i in range(chunksize):
        print('Reading: {}'.format('matrizes/' + name + str(i) + '.json'))
        chunk = read_json('matrizes/' + name + str(i) + '.json')
        if result.any():
            result = np.concatenate((result, chunk))
        else:
            result = chunk
    return result

def read_json(file_name):
    datafile = open(file_name, 'r')
    data = datafile.read()
    datafile.close()
    del datafile
    return np.array(json.loads(data))

