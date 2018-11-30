#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diverse useful functions for the orca prediction framework in the domain of
conditioal reasoning.
Copyright 2018 Cognitive Computation Lab
University of Freiburg
Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""

import numpy as np


def one_hot(data, length=16):
    '''
    Transform data of form [1, 1, 1, 1] to one-hot-encoding form [0, 0, .... 1]
    >>> one_hot(np.array([0, 0]), length=4).tolist()
    [[1.0, 0.0, 0.0, 0.0]]
    '''

    powers = np.array([2 ** i for i in range(int(np.sqrt(length)) - 1, -1, -1)])
    if data.ndim == 1:
        indices = [(data * powers).sum()]
    elif data.ndim == 2:
        indices = (data * powers).sum(axis=1)
    else:
        raise ValueError('Cannot handle an array of dim > 2')

    n_rows = len(indices)
    one_hot_data = np.zeros((n_rows, length))

    for i in range(n_rows):
        one_hot_data[i, indices[i]] = 1
    return one_hot_data


class ModelBase(object):
    """Base class for (conditional) model implementations """

    def __init__(self, model_name='ModelBase'):
        self.__name__ = model_name

    def fit(self, dataset=None, loss_function=None, initial_parameters=None):
        '''Computes the model's prediction to a syllogistic problem.'''
        pass

    def predict(self, problem):
        '''Computes the model's prediction to a syllogistic problem.'''
        raise NotImplementedError("Fitting method not implemented")

    def feedback(self, problem, answer):
        '''Tells the model the actual answer it predicted before.'''
        pass
