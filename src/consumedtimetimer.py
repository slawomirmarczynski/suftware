# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:22:58 2021

A class supporting measurements of the compute time consumption.

@author: Sławomir Marczyński
"""

import sys
import time


class ConsumedTimeTimer:
    """
    An adapter/facade/strategy to timer functions for various Pythons.

    Since Python 3.8 the old gold time.clock() method does not work. The method
    had been marked as deprecated since Python 3.3. After the update/upgrade to
    Python 3.8 a call of time.clock() raises an AttributeError exception
    the message "module 'time' has no attribute 'clock'". This make some older
    code forward incompatible. The simple solution would be replacing
    time.clock() with time.process_time(). Nethertheless, this may lead to
    backward incompatible code. Therefore, in order to overcome these
    difficulties, the idea was born to create ConsumedTimeTimer class as
    an object-oriented facade that simplifies the choice of strategy for
    selecting the appropriate time function.
    """

    def __init__(self, exclude_sleep_time=True):
        """
        Initialize an ConsumedTime object.

        Args:
            exclude_sleep: allows to select whether to measure only
                           the actively spent time (time used by the process),
                           or also the inactivity time (i.e. wall time).
                           Defaults to True.
        Returns:
            None.

        """
        self._last_tic_time = 0  # to avoid an undefined behaviour

        if sys.version_info.major <= 3 and sys.version_info.minor <= 3:
            self.get_time = time.clock  # pylint: disable=no-member
        elif not exclude_sleep_time:
            self.get_time = time.perf_counter
        else:
            self.get_time = time.process_time

    def __call__(self):
        """
        An redefinition of the call operator. This makes using ConsumedTime
        objects very easy, see example below::

            timer = ConsumedTimeTimer()
            t1 = timer()
            print('currently timer shows', t1, 'seconds')
            t2 = timer()
            print(t2 - t1, 'seconds missed from previous message')

        Notice, that t1 is not (maybe not) equal 0. This is because objects
        ConsumedTime neither reset clock nor remeber a time offset.

        Returns:
            float: the time as a float point number of seconds.

        """
        return self.get_time()

    def tic(self):
        """
        An method like Matlab toc - use it to start measuring computation
        time::

            timer = ConsumedTimeTimer()
            timer.tic()
            ...
            consumed_time = timer.toc()

        Returns:
            float: time, in seconds, between the call of tic and the call toc
                   methods.
        """
        self._tic = self()
        return self._tic

    def toc(self):
        delta = self() - self._last_tic_time
        # Remove comment to print messages like Matlab do.
        # print('Elapsed time is', delta, 'seconds.')
        return delta
