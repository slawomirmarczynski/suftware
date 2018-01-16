#!/usr/bin/env python

execfile('test_header.py')

TestCase(
    'Test that DEFT works on a simple data set',
    N=10000, should_succeed=True).run()
