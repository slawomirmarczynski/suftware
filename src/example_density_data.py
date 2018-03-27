#!/usr/bin/python
import numpy as np
from src.utils import DeftError
import os

# Load directory of file
data_dir = os.path.dirname(os.path.abspath(__file__))+'/../examples/data'

# List of supported distributions by name
VALID_DATASETS = ['.'.join(name.split('.')[:-1]) for name in
                  os.listdir(data_dir) if '.txt' in name ]


def parse_dataset(file_name):
    # Load data
    data = np.genfromtxt(file_name)

    # Fill in details from data file header
    details = {}
    header_lines = [line.strip()[1:] for line in open(file_name, 'r')
                    if line.strip()[0] == '#']
    for line in header_lines:
        key = eval(line.split(':')[0])
        value = eval(line.split(':')[1])
        details[key] = value

    return data, details


def run(dataset='old_faithful_eruption_times'):
    """
    Returns data from a predefined dataset

    Args:
        - dataset (str): Name of the dataset to load

    Returns:
        - data (numpy.array): An array containing sampled data
        - details (np.array, optional): Optional return value containing
            meta information
    """


    # Load data
    if dataset in VALID_DATASETS:
        # Set file name
        file_name = '%s/%s.txt'%(data_dir, dataset)

        # Load data
        data, details = parse_dataset(file_name)

    else:
        raise DeftError('Distribution type "%s" not recognized.'%distribution_type)

    return data, details


