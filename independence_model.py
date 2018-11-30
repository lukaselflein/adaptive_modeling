#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional Reasoning Baseline model
"""
from chater_oaksford_model import ChaterOaksford


class IndependenceModel(ChaterOaksford):
    """
    Chater and Oaksfords Independence Model for the Wason Selection TaskY
    """

    def __init__(self, model_name='IndependenceModel'):
        super(IndependenceModel, self).__init__(model_name=model_name)

    @staticmethod
    def model(p_antecedent, p_consequent, *args):
        # pylint: disable=unused-argument
        '''
        Return a function that gives the model predictions when parametrized.
        Parameters: a=P(p), b=P(q), e=p(not q|p).
        >>> i = IndependenceModel()
        >>> i.model(1, 1)
        [1, 0, 1, 0]
        '''
        endorsements = []
        # implicit ordering is MP, MT, AC, DA
        # MP
        endorsements += [p_consequent]
        # MT
        endorsements += [(1 - p_antecedent)]
        # AC
        endorsements += [p_antecedent]
        # DA
        endorsements += [(1 - p_consequent)]
        return endorsements
