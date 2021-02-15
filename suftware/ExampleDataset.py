import numpy as np

class ExampleDataset:
    """    
    """

    def __init__(self, file_name):
        self.data = np.genfromtxt(file_name)
