#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A uniform baseline model for conditional reasoning
"""

import utility


class UniformModel(utility.ModelBase):
    '''
    Baseline model which uniformely predicts
    all possible answers for conditional reasoning
    >>> u = UniformModel()
    >>> u.predict('MP')
    0.5
    >>> u.predict('MT')
    0.5
    '''

    def __init__(self, model_name='UniformModel'):
        '''Set a name for the model'''
        # self.__name__ = model_name
        super(UniformModel, self).__init__(model_name=model_name)

    def predict(self, problem):
        '''
        Predicts that every answer is given with equal probablity.
        '''
        prediction = 0.5
        return prediction
