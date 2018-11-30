#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of scoring rules for probabilistic prediction.
Copyright 2018 Cognitive Computation Lab
University of Freiburg
Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""

import numpy as np
import scipy.stats


def rmse(predictions, target_data):
    '''Returns Root Mean Squared Error between prediction and target data'''
    return np.sqrt(np.mean((predictions - target_data)**2))


def calc_rmse(parameters, model, target_data):
    '''Calculate Root Mean Squared Error between parameterized model and target data.'''
    predictions = model(*parameters)
    error = rmse(predictions, target_data)
    return error


def check_input(prediction, truth):
    '''
    Check format and validity of input.
    Predictions must be in format np.array([[0.1, 0.2, 0.3, 0.4]]).
    Truth must be in format np.array([[1.0, 0., 0., 0.]]).
    Truth may only contain one true result per row.
    Both Truth and Prediction rows must sum to one.
    '''
    if prediction.shape != truth.shape:
        raise ValueError('Input Dimension mismatch: prediction.shape = {}, tr\
                          uth.shape {}'.format(prediction.shape, truth.shape))
    if any(prediction.sum(axis=1) > 1.001) or \
       any(prediction.sum(axis=1) < 0.999) or \
       any(truth.sum(axis=1) > 1.001) or \
       any(truth.sum(axis=1) < 0.999):
        raise ValueError('Unnormalized input: prediction.sum = {},\
                          truth.sum {}'.format(prediction.sum(axis=1),
                                               truth.sum(axis=1)))
    if np.count_nonzero(truth) > truth.shape[0]:
        raise ValueError('Invalid input: \ntruth = \n {}\
                          \nMust be in format [0,..., 1, ... 0]'.format(truth))


def log_score(prediction, truth):
    '''
    Takes a matrix of predictions and actual outcomes, returns the logarithmic
    score.
    Outcomes or ground truth is encoded as rows containing one 1 and zeros.
    Predictions are encoded as probablilities in [0, 1], with normalized rows.
    The logcsore is a strictly proper local scoring rule.
    It minimizes expected surprise.
    >>> truth = np.array([[1, 0, 0]] * 2)
    >>> prediction = np.array([[0.3, 0.3, 0.4]] * 2)
    >>> int(log_score(prediction, truth) * 100)
    -240
    '''
    # Check input validity and format
    check_input(prediction, truth)

    # Get the matrix elements that came out true
    outcome_predictions = prediction[np.where(truth > 0.99)]
    # The score is the sum of logarithms of the predictions
    score = np.sum(np.log(outcome_predictions))
    return score


def mean_log_score(prediction, truth):
    '''
    Averaged log score.
    >>> truth = np.array([[1, 0, 0]] * 2)
    >>> prediction = np.array([[0.3, 0.3, 0.4]] * 2)
    >>> int(mean_log_score(prediction, truth) * 100)
    -120
    '''
    score = log_score(prediction, truth)
    return score / truth.shape[0]


def quad_score(prediction, truth):
    '''
    Takes a matrix of predictions and actual outcomes.
    Returns the qudratic Brier-score.
    Outcomes or ground truth is encoded as rows containing one 1 and zeros.
    Predictions are encoded as probablilities in [0, 1], with normalized rows.
    The quadritic score (alias Brierscore) is a proper scoring rule.
    It is equivalent to the mean squared error of the prediction.
    >>> truth = np.array([[1, 0, 0]] * 2)
    >>> prediction = np.array([[0.3, 0.3, 0.4]] * 2)
    >>> int(quad_score(prediction, truth) * 100)
    74
    '''
    # Check input validity and format
    check_input(prediction, truth)

    # The score is the sum of squared deviations
    score = np.sum((truth - prediction)**2)
    # Divided by the number of forecasts for normalization
    score /= truth.shape[0]
    return score


def accuracy(prediction, truth):
    '''
    Takes a matrix of predictions and actual outcomes.
    Returns the accuracy at rank 1.
    Outcomes or ground truth is encoded as rows containing one 1 and zeros.
    Predictions are encoded as probablilities in [0, 1], with normalized rows.
    The accuracy is an *improper* scoring rule.
    Also, it is stricly positive, better prediciton produce greater numbers.
    >>> truth = np.array([[1, 0, 0],[0,1,0]])
    >>> prediction = np.array([[0.3, 0.3, 0.4]] * 2)
    >>> int(accuracy(prediction, truth))
    0
    >>> prediction = np.array([[0.4, 0.5, 0.1]] * 2)
    >>> int(accuracy(prediction, truth))
    1
    >>> prediction = np.array([[0.4, 0.3, 0.3]] * 2)
    >>> int(accuracy(prediction, truth))
    1
    '''
    # Check input validity and format
    check_input(prediction, truth)

    matrix = prediction.copy()
    # Only maximum value predictions are considered
    for row in range(0, truth.shape[0]):
        mask = abs(prediction[row] - np.max(prediction[row])) < 0.001
        # If multiple entries have first rank, one is chosen uniformely.
        # This is equivalent to giving a score of 1/N
        matrix[row][mask] = 1 / len(prediction[row][mask])
        # All other predictions are neglected
        matrix[row][np.logical_not(mask)] = 0

    # Only count the predictions of the actual outcomes
    matrix = matrix[np.where(truth > 0.99)]
    score = np.sum(matrix)
    return score


def mean_accuracy(prediction, truth):
    '''
    Averages the accuracy score.
    >>> truth = np.array([[1, 0, 0]] * 2)
    >>> prediction = np.array([[0.4, 0.3, 0.3]] * 2)
    >>> mean_accuracy(prediction, truth)
    1.0
    '''
    score = accuracy(prediction, truth)
    return score / truth.shape[0]


def mrr_score(prediction, truth):
    '''
    Takes a matrix of predictions and actual outcomes.
    Returns the accuracy at rank 1.
    Outcomes or ground truth is encoded as rows containing one 1 and zeros.
    Predictions are encoded as probablilities in [0, 1], with normalized rows.
    Scores a prediction with the inverse of the predicted rank.
    In case of ties, the maximum rank is given to both contestants.
    This is an *improper* scoring rule.
    Also, it is stricly positive, better prediciton produce greater numbers.
    >>> truth = np.array([[1, 0, 0]] * 2)
    >>> prediction = np.array([[0.3, 0.3, 0.4]] * 2)
    >>> int(mrr_score(prediction, truth)*100)
    33
    >>> prediction = np.array([[0.4, 0.3, 0.3]] * 2)
    >>> int(mrr_score(prediction, truth)*100)
    100
    '''
    # Check input validity and format
    check_input(prediction, truth)

    # Invert probablities as small numbers are ranked higher
    ranked = 1 / prediction.copy()
    # Rank predictions
    for row in range(ranked.shape[0]):
        # The max rank is given in case of ties
        ranked[row] = scipy.stats.rankdata(ranked[row], method='max')

    truth_mask = np.where(truth > 0.99)
    # The score is 1/maxrank
    ranked_scores = 1 / ranked[truth_mask]
    score = np.sum(ranked_scores)
    return score / truth.shape[0]
