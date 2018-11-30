"""
A statistical model for conditional reasoning based on a paper by Evans (1977)
"""
import numpy as np


class Evans(object):
    '''
    The model describes probabilities of selection.
       We have four possible rules:
           If p then q
           If p then not q
           If not p then q
           If not p then not q
       For each of the rules, there are the following selections
       (a participant can select more than one):
           p, not p, q, not q

       General theoretical expression of response probability:

       Pr(r) = alpha * I + (1 - alpha) * R

       where r ranges over p, not p, q, not q
       alpha (weighting factor) can be alpha_a for antecedent choice
           and alpha_c for consequent choice
       I (interpretational factor) has 1 for logically correct response, 0 for incorrect
           (if given 'If p then q' correct responses are p and not q, incorrect: not p and q)
       R (response factor) can be Rmat for match, Rmis for mismatch
           (if given 'If p then q' matches are p and q, mismatches are not p and not q)
    '''

    def __init__(self, model_name='evans'):
        '''
        Initialize the data structure for the parameters
        '''
        # order: [alpha_a, alpha_c, Rmat, Rmis]
        self.__name__ = model_name
        self.parameters = np.zeros(4)

    def model(self, rule):
        '''
        Returns the probabilities of accepting the inferences.

        We make an assumption that based on the selection,
            the participant accepts a certain inference...
        e.g. If rule is 'If p then q' we have the following possible selections
            and what they correspond to:
            - selection p -- accepting MP
            - selection not p -- accepting DA
            - selection q -- accepting AC
            - selection not q -- accepting MT
        The two logically correct acceptances in this case would be MP and MT.
        Similarly, the same is valid for the other rules.

        rule: encoded type of rule
            0 - If p then q
            1 - If p then not q
            2 - If not p then q
            3 - If not p then not q
        returns: probabilities
        '''

        # Probabilities, ordering [MP, MT, AC, DA]
        probs = np.zeros(4)

        # If p then q
        if rule == 0:
            probs[0] = self.parameters[0] + (1 - self.parameters[0]) * self.parameters[2]  # p - MP
            probs[3] = (1 - self.parameters[0]) * self.parameters[3]  # np - DA
            probs[2] = (1 - self.parameters[1]) * self.parameters[2]  # q - AC
            probs[1] = self.parameters[1] + (1 - self.parameters[1]) * self.parameters[3]  # nq - MT
        # If p then not q
        elif rule == 1:
            probs[0] = self.parameters[0] + (1 - self.parameters[0]) * self.parameters[2]  # p - MP
            probs[3] = (1 - self.parameters[0]) * self.parameters[3]  # np - DA
            probs[1] = self.parameters[1] + (1 - self.parameters[1]) * self.parameters[2]  # q - MT
            probs[2] = (1 - self.parameters[1]) * self.parameters[3]  # nq - AC
        # If not p then q
        elif rule == 2:
            probs[3] = (1 - self.parameters[0]) * self.parameters[2]  # p - DA
            probs[0] = self.parameters[0] + (1 - self.parameters[0]) * self.parameters[3]  # np - MP
            probs[2] = (1 - self.parameters[1]) * self.parameters[2]  # q - AC
            probs[1] = self.parameters[1] + (1 - self.parameters[1]) * self.parameters[3]  # nq - MT
        # If not p then not q
        elif rule == 3:
            probs[3] = (1 - self.parameters[0]) * self.parameters[2]  # p - DA
            probs[0] = self.parameters[0] + (1 - self.parameters[0]) * self.parameters[3]  # np - MP
            probs[1] = self.parameters[1] + (1 - self.parameters[1]) * self.parameters[2]  # q - MT
            probs[2] = (1 - self.parameters[1]) * self.parameters[3]  # nq - AC
        else:
            raise ValueError('rule must have a value 0, 1, 2, or 3')

        return probs

    @staticmethod
    def calculate_alpha(prob1, prob2):
        '''
        Helper function for the fitting method,
        estimates the alpha parameter given two probability values, prob1 and prob2.

        The expressions for prob1 and prob2 are of the form:
        prob1 = alpha + (1 - alpha) * R
        prob2 = (1 - alpha) * R

        alpha = prob1 - prob2
        '''

        return prob1 - prob2

    @staticmethod
    def calculate_r(prob, est_alpha):
        '''
        Helper function for the fitting method,
        estimates the R parameter given a probability value, prob,
        and est_alpha being the estimation for the corresponding alpha.

        The expressions for prob and est_alpha are of the form:
        prob = (1 - alpha) * R
        est_alpha = alpha

        R = prob / (1 - est_alpha)
        '''

        return prob / (1 - est_alpha)

    def fit(self, probabilities):
        '''
        Given experimental data, estimate parameters alpha_a, alpha_c, Rmat and Rmis.

        The argument 'probabilities' has 16 probability values -
            one for each of the four possible selections for each
            of the four types of rules, obtained from aggregated data.

        Notation:
        xn - selection of x, given rule n
        x = [p, np, q, nq]
        n = [0 (p -> q), 1 (p -> nq), 2 (np -> q), 3 (np -> nq)]
        e.g.:
        Given rule 0 (If p then q), the expression for a participant selecting p is denoted by p0.

        All expressions:
        Antedecent choices:
        p0 = p1 = alpha_a + (1 - alpha_a) * Rmat
        p2 = p3 = (1 - alpha_a) *  Rmat
        np2 = np3 = alpha_a + (1 - alpha_a) * Rmis
        np0 = np1 = (1 - alpha_a) * Rmis

        Consequent choices:
        q1 = q3 = alpha_c + (1 - alpha_c) * Rmat
        q0 = q2 = (1 - alpha_c) * Rmat
        nq0 = nq2 = alpha_c + (1 - alpha_c) * Rmis
        nq1 = nq3 = (a - alpha_c) * Rmis

        The estimates are obtained by solving a system of equations
            (the expressions) for the parameter.
        e.g. Estimation of alpha_a:
            - It is contained in all expressions for antedecent choices
            - Average the pairs of probabilities which are described by the same expression
                (the pairs here: (p0, p1), (p2, p3), (np2, np3), (np0, np1))
            - First estimate of alpha_a obtained
                from the average of p2 and p3 and the average of p0 and p1
            - Second estimate of alpha_a obtained
                from the average of np0 and np1 and the average of np2 and np3
            - The final estimate of alpha_a is set to be the mean of these two estimates
        The other parameters are estimated similarly.

        probabilities format: array
        '''

        # Estimated alpha_a
        self.parameters[0] = np.mean([[self.calculate_alpha(
            np.mean([probabilities[0], probabilities[4]]),
            np.mean([probabilities[8], probabilities[12]]))],
            [self.calculate_alpha(
                np.mean([probabilities[9], probabilities[13]]),
                np.mean([probabilities[1], probabilities[5]]))]])

        # Estimated alpha_c
        self.parameters[1] = np.mean([[self.calculate_alpha(
            np.mean([probabilities[6], probabilities[14]]),
            np.mean([probabilities[2], probabilities[10]]))],
            [self.calculate_alpha(
                np.mean([probabilities[3], probabilities[11]]),
                np.mean([probabilities[7], probabilities[15]]))]])

        # Estimated Rmat
        self.parameters[2] = np.mean([[self.calculate_r(
            np.mean([probabilities[8], probabilities[12]]), self.parameters[0])],
            [self.calculate_r(
                np.mean([probabilities[2], probabilities[10]]),
                self.parameters[1])]])

        # Estimated Rmis
        self.parameters[3] = np.mean([[self.calculate_r(
            np.mean([probabilities[1], probabilities[5]]), self.parameters[0])],
            [self.calculate_r(
                np.mean([probabilities[7], probabilities[15]]),
                self.parameters[1])]])

    def predict(self, rule, problem):
        '''
        Predicts probabilities for every possible answer, depending on the form of the rule.

        The probability is calculated by multiplying probabilities obtained by the model,
         e.g. if we are interested in the probability of accepting both MP and MT we multiply
         the probability of MP and the probability of MT.

        rule: encoded type of rule
            0 - If p then q
            1 - If p then not q
            2 - If not p then q
            3 - If not p then not q
        problem: array consisting of the inference forms
        for which we want the probability of them being accepted
            (e.g. ['MP', 'MT'])
        returns: probabilities
        '''

        predictions = dict()
        predictions['MP'] = self.model(rule)[0]
        predictions['MT'] = self.model(rule)[1]
        predictions['AC'] = self.model(rule)[2]
        predictions['DA'] = self.model(rule)[3]

        res = 1.0
        for form in problem:
            res = res * predictions[form]

        return res

    def test(self):
        '''
        Fitting the model based on the observed probabilities given
        in the paper and comparing the obtained estimated parameters
        to the one in the paper.
        '''
        # Observed probabilities (Table II)
        obs_prob = np.zeros(16)
        # If p then q
        obs_prob[0] = 0.875  # p0
        obs_prob[1] = 0.083  # np0
        obs_prob[2] = 0.500  # q0
        obs_prob[3] = 0.333  # nq0
        # If p then not q
        obs_prob[4] = 0.917  # p1
        obs_prob[5] = 0.042  # np1
        obs_prob[6] = 0.583  # q1
        obs_prob[7] = 0.083  # nq1
        # If not p then q
        obs_prob[8] = 0.292  # p2
        obs_prob[9] = 0.583  # np2
        obs_prob[10] = 0.583  # q2
        obs_prob[11] = 0.417  # nq2
        # If not p then not q
        obs_prob[12] = 0.458  # p3
        obs_prob[13] = 0.542  # np3
        obs_prob[14] = 0.750  # q3
        obs_prob[15] = 0.292  # nq3

        self.fit(obs_prob)

        # Comparison of estimated parameters here, and in the paper
        paper_parameters = [0.511, 0.156, 0.704, 0.176]
        print("Estimated parameters vs. Parameters in paper: ")
        print("alpha_a: {} {}".format(self.parameters[0], paper_parameters[0]))
        print("alpha_c: {} {}".format(self.parameters[1], paper_parameters[1]))
        print("Rmat: {} {}".format(self.parameters[2], paper_parameters[2]))
        print("Rmis: {} {}".format(self.parameters[3], paper_parameters[3]))

        # Examples of predicting probabilities
        print("Given rule 'If p then q', probability of accepting "
              "MP: {}".format(self.predict(0, ['MP'])))
        print("Given rule 'If p then not q', probability of accepting "
              "MP and AC: {}".format(self.predict(1, ['MP', 'AC'])))
        print("Given rule 'If not p then q', probability of accepting "
              "MP and DA: {}".format(self.predict(2, ['MP', 'DA'])))
        print("Given rule 'If not p then not q', probability of accepting "
              "MP, MT and AC: {}".format(self.predict(3, ['MP', 'MT', 'AC'])))
