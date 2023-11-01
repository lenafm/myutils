# manage file systems

import os
import pickle


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

