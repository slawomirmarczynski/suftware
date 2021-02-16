"""
"""

import utils
from utils import ControlledError
from utils import enable_graphics, check

def demo(example='real_data'):
    """
    Performs a demonstration of suftware.

    Parameters
    ----------

    example: (str)
        A string specifying which demo to run. Must be 'real_data',
        'simulated_data', or 'custom_data'.

    Return
    ------

    None.
    """

    import os
    example_dir = os.path.dirname(__file__)

    example_dict = {
        'custom_data': 'docs/example_custom.py',
        'simulated_data': 'docs/example_wide.py',
        'real_data': 'docs/example_alcohol.py'
    }

    check(example in example_dict,
          'example = %s is not valid. Must be one of %s'%\
          (example, example_dict.keys()))

    file_name = '%s/%s'%(example_dir, example_dict[example])
    with open(file_name, 'r') as f:
        content = f.read()
        line = '-------------------------------------------------------------'
        print('Running %s:\n%s\n%s\n%s'%\
              (file_name, line, content, line))
    exec(open(file_name).read())
