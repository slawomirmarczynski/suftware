#!/usr/bin/python
import numpy as np
from src.utils import ControlledError
import os

# Load directory of file
data_dir = os.path.dirname(os.path.abspath(__file__))+'/../examples/data'

# List of supported distributions by name
VALID_DATASETS = ['.'.join(name.split('.')[:-1]) for name in
                  os.listdir(data_dir) if '.txt' in name]
VALID_DATASETS.sort()

class ExampleData:
    """
    Provides an interface to example data provided with the SUFTware package.

    Parameters
    ----------

    dataset: (str)
        Name of dataset to load. Run sw.ExampleData.list() to see
        which datasets are available.

    """


    def __init__(self, dataset='old_faithful_eruption_times'):
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
            # Set file dataset
            file_name = '%s/%s.txt'%(data_dir, dataset)

            # Load data
            self._load_dataset(file_name)

        else:
            raise ControlledError('Distribution type "%s" not recognized.' %
                                  dataset)


    def _load_dataset(self, file_name):
        # Load data
        self.data = np.genfromtxt(file_name)

        # Fill in details from data file header
        details = {}
        header_lines = [line.strip()[1:] for line in open(file_name, 'r')
                        if line.strip()[0] == '#']
        for line in header_lines:
            key = eval(line.split(':')[0])
            value = eval(line.split(':')[1])
            try:
                setattr(self, key, value)
            except:
                ControlledError('Error loading example data. Either key or value'
                          'of metadata is invalid. key = %s, value = %s' %
                                (key, value))

    @staticmethod
    def list():
        """
        Return list of available datasets.
        """
        return VALID_DATASETS


