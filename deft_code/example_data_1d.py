#!/usr/bin/python
import argparse
import json
import numpy as np
import math
import scipy as sp
import scipy.stats as stats
import sys
from deft_code.utils import DeftError
import os

# Load directory of file
data_dir = os.path.dirname(os.path.abspath(__file__))+'/../data'

# List of supported distributions by name
# VALID_DATASETS = '''
# buffalo_snowfall
# old_faithful_eruption_times
# old_faithful_waiting_times
# treatment_length
# kiva_loans__funded_amount
# kiva_loans__lender_count
# kiva_loans__term_in_months
# '''.split()
VALID_DATASETS = [name.split('.')[0] for name in
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


def run(dataset='old_faithful_eruption_times', return_details=False):
    """
    Returns data from a predefined dataset

    Args:
        - dataset (str): Name of the dataset to load
        - return_details (bool): Whether or not to return meta information
        about data

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

    if return_details:
        return data, details
    else:
        return data

