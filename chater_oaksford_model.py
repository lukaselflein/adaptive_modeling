#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A probablistic model for conditional reasoning based on a paper by Chater, Oaksford & Larkin (2000)
"""
import utility
import scoring_rules
import scipy.optimize
import numpy as np


class ChaterOaksford(utility.ModelBase):
    '''
    Subjective-probability based model by Chater,Oaksford & Larkin (2000)
    for conditional reasoning.
    The basic idea is that people have some subjective probablity assigned
    to the antecedent, consequent and the conditional rule being correct.
    These probablities can be combined to yield a probablity of accepting
    each of the four classical conditionals.
    Using these three probabilities as fitting parameters, empirical data
    can be modeled.
    '''

    def __init__(self, model_name='ChaterOaksford'):
        '''
        Define a name for the model, initialize some data structures
        '''
        super(ChaterOaksford, self).__init__(model_name=model_name)
        self.subjective_probabilities = dict()
        self.aggregate_data = None

    @staticmethod
    def model(p_antecendent=0.5, p_consequent=0.5, p_exception=0.5):
        '''
        Returns the endorsements given the modus (MP, MT, AC, DA) and parameters.
        This is a probablilistic conditional reasoning model.
        See Oaksford, Chater, Lakin 2000

        Endorsement is the probability of the reasoner accepting the modus.
        E.g. the Modus Ponens is accepted with a probablitiy of
        1 - P(not q|p), which is just the probablity of the conditional rule
        being true.

        Parameters:
        p_antecdent  = P(p)       = a
        p_consequent = P(q)       = b
        p_exception  = P(not q|p) = e
        '''
        endorsements = np.zeros(4)
        # The implicit ordering of the conditional modi is [MP, MT, AC, DA]
        # MP: 1 - e
        endorsements[0] = 1 - p_exception

        # MT: (1 - b - a * e) / (1 - b)
        endorsements[1] = (1 - p_consequent - p_antecendent * p_exception)
        endorsements[1] /= (1 - p_consequent)

        # AC: (a * (1 - e)) / b
        endorsements[2] = (p_antecendent * (1 - p_exception)) / p_consequent

        # DA: (1 - b - a * e) / (1 - a)
        endorsements[3] = (1 - p_consequent - p_antecendent * p_exception)
        endorsements[3] /= (1 - p_antecendent)
        return endorsements

    def fit(self, dataset=None, loss_function=scoring_rules.calc_rmse,
            initial_parameters=(0.5, 0.5, 0.5)):
        '''
        Optimize parameters on aggregate date with a loss function
        and the 'model' routine.

        dataset format: pandas table with the modi as column names
        returns: set of optimized parameters
        '''
        if dataset is not None:
            self.aggregate_data = (dataset.sum() / dataset.shape[0] / 2).values
        elif self.aggregate_data is not None:
            pass
        else:
            raise ValueError('Neither dataset nor self.aggregate_data is defined')

        bounds = [(0.00001, 0.99999)] * len(initial_parameters)
        res = scipy.optimize.minimize(loss_function, initial_parameters,
                                      method='L-BFGS-B',
                                      bounds=bounds, options={'disp': False},
                                      args=(self.model, self.aggregate_data))

        self.subjective_probabilities['MP'] = self.model(*res.x)[0]
        self.subjective_probabilities['MT'] = self.model(*res.x)[1]
        self.subjective_probabilities['AC'] = self.model(*res.x)[2]
        self.subjective_probabilities['DA'] = self.model(*res.x)[3]
        return {'a': res.x[0], 'b': res.x[1], 'e': res.x[2]}

    def predict(self, problem):
        '''
        Predicts every answer with population level frequency.

        problem format: 2-letter string, e.g. 'MP'
        returns: float probablity of responding yes to the problem
        '''
        prediction = self.subjective_probabilities[problem]
        return prediction
