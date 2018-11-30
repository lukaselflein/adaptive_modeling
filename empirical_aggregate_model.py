#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional Reasoning Baseline model
"""
import utility


class EmpiricalAggregate(utility.ModelBase):
    '''Baseline model for conditional reasoning.
       Predictions are equal to population mean answer probablities'''

    def __init__(self, model_name='EmpiricalAggregate'):
        '''Set a name for the model'''
        super(EmpiricalAggregate, self).__init__(model_name=model_name)
        self.empirical_frequency = dict()

    def fit(self, dataset):
        self.empirical_frequency['MP'] = dataset['MP'].mean() / 2
        self.empirical_frequency['MT'] = dataset['MT'].mean() / 2
        self.empirical_frequency['AC'] = dataset['AC'].mean() / 2
        self.empirical_frequency['DA'] = dataset['DA'].mean() / 2

    def predict(self, problem):
        '''Predicts every answer with population level frequency.'''
        prediction = self.empirical_frequency[problem]
        return prediction
