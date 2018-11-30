#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Conditional Reasoning model
"""
from chater_oaksford_model import ChaterOaksford
import scoring_rules
import scipy.optimize


class AdaptiveChaterOaksford(ChaterOaksford):
    '''Subjective-probability based model by Chater&Oaksford 2000 for conditional reasoning'''

    def __init__(self, model_name='AdaptiveChaterOaksford', weight=1):
        '''Set a name for the model'''
        super(AdaptiveChaterOaksford, self).__init__(model_name=model_name)
        self.new_data_weight = weight

    def fit(self, dataset=None, loss_function=scoring_rules.calc_rmse,
            initial_parameters=(0.5, 0.5, 0.5)):
        '''Optimize parameters on aggregate date with a loss function and the 'model' routine '''
        if dataset is not None:
            self.aggregate_data = (dataset.sum() / dataset.shape[0] / 2).values
        elif self.aggregate_data is not None:
            pass
        else:
            raise ValueError('Neither dataset nor self.aggregate_data is defined')

        bounds = [(0.00001, 0.99999)] * len(initial_parameters)
        res = scipy.optimize.minimize(loss_function, initial_parameters, method='L-BFGS-B',
                                      bounds=bounds, options={'disp': False},
                                      args=(self.model, self.aggregate_data))

        self.subjective_probabilities = dict()
        self.subjective_probabilities['MP'] = self.model(*res.x)[0]
        self.subjective_probabilities['MT'] = self.model(*res.x)[1]
        self.subjective_probabilities['AC'] = self.model(*res.x)[2]
        self.subjective_probabilities['DA'] = self.model(*res.x)[3]
        return {'a': res.x[0], 'b': res.x[1], 'e': res.x[2]}

    def feedback(self, problem='MT', answer=1):
        '''Saves participant's answers into internal data structure.'''
        self.adapt(problem, answer)

    def adapt(self, problem, answer, weight=0):
        '''Adapts model predictions to incorporate information on the participant at hand.'''
        weight = self.new_data_weight
        ordering = ['MP', 'MT', 'AC', 'DA']
        self.aggregate_data[ordering.index(problem)] += answer * weight
        self.aggregate_data[ordering.index(problem)] /= weight + 1
        self.fit()
