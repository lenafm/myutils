# manage file systems

import os
import csv
import pickle
import numpy as np


def get_dir(indir, filename=None):
    if filename is None:
        return os.path.abspath(os.path.join('..', indir))
    else:
        return os.path.abspath(os.path.join('..', indir, filename))


def open_pickle_list(ls):
    pickles = []
    for file in ls:
        pickles.append(load_pickle(file))
    return pickles


def load_pickle(filename):
    with open(filename, 'rb') as f:
        pickledfile = pickle.load(f)
    return pickledfile


def write_pickle(file, filename):
    with open(filename, 'wb') as f:
        pickle.dump(file, f)


def write_list_to_csv(ls, filename, header=None):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if header is not None:
            if isinstance(ls[0], (tuple, list, np.ndarray)):
                assert len(header) == len(ls[0])
            else:
                assert isinstance(header, list)
                assert len(header) == 1
            writer.writerow(header)
        if isinstance(ls[0], (tuple, list, np.ndarray)):
            writer.writerows(ls)
        else:
            writer.writerows((item,) for item in ls)


def get_list_from_csv(file, datatype):
    ls = []
    with open(file) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            try:
                if datatype == 'string':
                    ls_element = str(row[0])
                elif datatype == 'int':
                    ls_element = int(row[0])
                elif datatype == 'float':
                    ls_element = float(row[0])
                else:
                    raise ValueError('Datatype needs to be "string", "int", or "float".')
                ls.append(ls_element)
            except ValueError:
                pass
    return ls


def get_list_of_tuples_from_csv(file, delim=','):
    ls = []
    with open(file) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delim)
        for row in csv_reader:
            try:
                ls.append(row)
            except ValueError:
                pass
    return ls

