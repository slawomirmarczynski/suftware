#!/usr/bin/env python

execfile('test_header.py')

TestCase(
    'Test that DEFT works on a simple data set',
    should_succeed=True).run()

TestCase(
    'Test that bbox of size > 2 fails',
    should_succeed=False, bbox=[-6,6,1]).run()

TestCase(
    'Test if bbox with misordered boundaries fails',
    should_succeed=False, bbox=[6,-6]).run()

TestCase(
    'Test that negative alpha leads to failure',
    should_succeed=False, alpha=-3).run()

TestCase(
    'Test that negative G leads to failure',
    should_succeed=False, G=-100).run()

TestCase(
    'Test if G=0 fails',
    should_succeed=False, G=0).run()

TestCase(
    'Test if non-integner G fails',
    should_succeed=False, G=100.5).run()

