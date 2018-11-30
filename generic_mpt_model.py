#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional Reasoning Baseline model
"""
import utility
import scoring_rules
import scipy.optimize
import numpy as np


class GenericMPT(utility.ModelBase):
    '''The simplest 3-parameter model for a 4-category prediction'''

    def __init__(self, model_name='GenericMPT'):
        '''Set a name for the model'''
        self.subjective_probabilities = dict()
        super(GenericMPT, self).__init__(model_name=model_name)

    def fit(self, dataset=None, loss_function=scoring_rules.calc_rmse,
            initial_parameters=(0.5, 0.5, 0.5)):
        '''Optimize parameters on aggregate date with a loss function and the 'model' routine '''
        aggregate_data = (dataset.sum() / dataset.shape[0] / 2).values
        bounds = [(0.00001, 0.99999)] * len(initial_parameters)
        res = scipy.optimize.minimize(loss_function, initial_parameters, method='L-BFGS-B',
                                      bounds=bounds, options={'disp': False},
                                      args=(self.model, aggregate_data))

        self.subjective_probabilities['MP'] = self.model(*res.x)[0]
        self.subjective_probabilities['MT'] = self.model(*res.x)[1]
        self.subjective_probabilities['AC'] = self.model(*res.x)[2]
        self.subjective_probabilities['DA'] = self.model(*res.x)[3]

    @staticmethod
    def model(*parameters):
        '''
        The model is the simplest mpt for four categories:
               a
           b        c
        |1| |2|  |3| |4|
        '''
        predictions = np.zeros(4)
        predictions[0] = parameters[0] * parameters[1]
        predictions[1] = parameters[0] * (1 - parameters[1])
        predictions[2] = (1 - parameters[0]) * parameters[2]
        predictions[3] = (1 - parameters[0]) * (1 - parameters[2])
        return predictions

    def predict(self, problem):
        '''Predicts every answer with population level frequency.'''
        prediction = self.subjective_probabilities[problem]
        return prediction
