"""
----------------------------------------------------
 Authors: A. Ramirez-Morales and J. Salmon-Gamboa
----------------------------------------------------
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class BoostedSVM:
    """
    Class that boost a SVM following the AdaBoost recipe
    """

    def __init__(
            self,
            C,
            gammaEnd,
            myKernel,
            myDegree=1,
            myCoef0=1,
            Diversity=False,
            early_stop=False,
            debug=False,
            train_verbose=False):
        self.C = C
        self.gammaEnd = gammaEnd
        self.myKernel = myKernel
        self.myDegree = myDegree
        self.myCoef0 = myCoef0
        self.weak_svm = ([])
        self.alphas = ([])
        self.weights_list = []
        self.errors = ([])
        self.precision = ([])
        self.train_scores = ([])
        self.test_scores = ([])
        self.count_over_train = ([])
        self.count_over_train_equal = ([])
        # Diversity threshold-constant and empty list
        self.m_div_flag = Diversity
        self.eta = 0.7
        self.diversities = ([])
        self.Div_total = ([])
        self.Div_partial = ([])
        self.debug = debug
        self.verbose_train = train_verbose
        self.early_flag = early_stop
        self.n_classifiers = 0

    def svc_train(
            self,
            myGamma,
            stepGamma,
            x_train,
            y_train,
            myWeights,
            count,
            flag_div,
            value_div):
        """
        Method to train a single classifier
        """
        if count == 0:
            myGamma = stepGamma

        while True:
            if myGamma > self.gammaEnd + stepGamma:
                return 0, 0, None, None
            errorOut = 0.0
            svcB = SVC(C=self.C,
                       kernel=self.myKernel,
                       degree=self.myDegree,
                       coef0=self.myCoef0,
                       gamma=myGamma,
                       shrinking=True,
                       probability=False,
                       tol=0.001,
                       cache_size=1000)
            svcB.fit(x_train, y_train, sample_weight=myWeights)
            y_pred = svcB.predict(x_train)
            # Error calculation
            for i in range(len(y_pred)):
                if (y_train[i] != y_pred[i]):
                    errorOut += myWeights[i]

            error_pass = errorOut < 0.499 and errorOut > 0.0
            # Diverse_AdaBoost, if Diversity=False, diversity plays no role in
            # classifier selection
            div_pass, tres = self.pass_diversity(
                flag_div, value_div, count, error_pass)
            if (error_pass and not div_pass):
                value_div = self.diversity(x_train, y_pred, count)
            if self.debug:
                print('error_flag: %5s | div_flag: %5s | div_value: %5s | Threshold: %5s | no. data: %5s | count: %5s | error: %5.2f | gamma: %5.2f | diversities  %3s '
                      % (error_pass, div_pass, value_div, tres, len(y_pred), count, errorOut, myGamma, len(self.diversities)))

            # Require an error below 50%, avoid null errors and diversity
            # requirement
            if (error_pass and div_pass):
                # myGamma -= stepGamma
                break

            myGamma += stepGamma

        return myGamma, errorOut, y_pred, svcB

    def fit(self, X, y):
        """
        Method that iterates the AdaBoost algorithm, it saves the individual classifiers and their errors
        """
        if self.early_flag:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, y, test_size=0.2)
            X_train, Y_train = self._check_X_y(X_train, Y_train)
            X_test, Y_test = self._check_X_y(X_test, Y_test)
        else:
            X_train, Y_train = self._check_X_y(X, y)

        n = X_train.shape[0]
        weights = np.ones(n) / n

        div_flag, div_value = self.m_div_flag, 0

        gammaMax = self.gammaEnd
        gammaStep, gammaVar = gammaMax / 100., 1 / 100.
        cost, count, norm = 1, 0, 0.0
        h_list = []

        # AdaBoost loop
        while True:
            if self.early_flag:
                if self.early_stop(count, X_test, Y_test, gammaVar):
                    break  # Early stop based on a score
            if count == 0:
                norm = 1.0
                new_weights = weights.copy()

            new_weights = new_weights / norm

            self.weights_list.append(new_weights)
            if self.debug:
                print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")

            # Call svm, weight samples, iterate sigma(gamma), get errors,
            # obtain predicted classifier (h as an array)
            gammaVar, error, h, learner = self.svc_train(
                gammaVar, gammaStep, X_train, Y_train, new_weights, count, div_flag, div_value)

            # Or learner == None or h == None):
            if (gammaVar > gammaMax or error <= 0):
                break

            # Count how many times SVM we add the ensemble
            count += 1
            self.n_classifiers += 1

            # Calculate training precision
            fp, tp = 0, 0
            for i in range(n):
                if (Y_train[i] != h[i]):
                    fp += 1
                else:
                    tp += 1

            # Store the predicted classes
            h_temp = h.tolist()
            h_list.append(h_temp)

            # Store errors and precision
            self.errors = np.append(self.errors, [error])
            self.precision = np.append(self.precision, [tp / (tp + fp)])

            # Calculate diversity
            if self.m_div_flag:
                div_value = self.diversity(
                    X_train, h, count)  # cf. h == y_pred

            # Classifier weights (alpha), obtain and store
            x = (1 - error) / error
            alpha = 0.5 * np.log(x)
            self.alphas = np.append(self.alphas, alpha)
            self.weak_svm = np.append(self.weak_svm, learner)

            # Get training errors used for early stop
            # train_score = 1 - accuracy_score(Y_train, self.predict(X_train))
            # self.train_scores = np.append(self.train_scores, train_score)

            # Reset weight lists
            weights = new_weights.copy()
            new_weights = ([])
            norm = 0.0
            # Set weights for next iteration
            for i in range(n):
                x = (-1.0) * alpha * Y_train[i] * h[i]
                new_weights = np.append(new_weights, [weights[i] * np.exp(x)])
                norm += weights[i] * np.exp(x)

            # Do loop as long gamma > gammaMin, if gamma < 0, SVM fails exit
            # loop
            if gammaVar >= gammaMax:  # ) or (gammaVar < 0):
                break
            # End of adaboost loop

        print(count, 'number of classifiers')
        self.n_classifiers = count
        if (count == 0):
            print(' WARNING: No selected classifiers in the ensemble!!!!')
            self.n_classifiers = 0
            return self

        # Show the training the performance (optional)
        if (self.verbose_train):
            h_list = np.array(h_list)
            # Calculate the final classifier
            h_alpha = np.array([h_list[i] * self.alphas[i]
                               for i in range(count)])
            # Final classifier is an array (size of number of data points)
            final = ([])
            for j in range(len(h_alpha[0])):
                suma = 0.0
                for i in range(count):
                    suma += h_alpha[i][j]
                    final = np.append(final, [np.sign(suma)])
                    # Final precision calculation
                    final_fp, final_tp = 0, 0
                    for i in range(n):
                        if (Y_train[i] != final[i]):
                            final_fp += 1
                        else:
                            final_tp += 1
                        final_precision = final_tp / (final_fp + final_tp)
            print(
                "Final training precision: {} ".format(
                    round(
                        final_precision,
                        4)))
        return self

    def predict(self, X):
        """
        This method makes predictions using already fitted model
        """
        # print(len(self.alphas), len(self.weak_svm), "how many alphas we have")
        # print(type(X.shape[0]), 'check size ada-boost')
        if self.n_classifiers == 0:
            return np.zeros(X.shape[0])
        svm_preds = np.array([learner.predict(X) for learner in self.weak_svm])
        return np.sign(np.dot(self.alphas, svm_preds))

    def diversity(self, x_train, y_pred, count):
        """
        This method gets div for a single classifier
        """
        if count == 1:
            # For 1st selected classifer, set max diversity
            return len(y_pred) / len(y_pred)
        div = 0
        # Uses the already selected classifiers in ensemble
        ensemble_pred = self.predict(x_train)
        for i in range(len(y_pred)):
            if (y_pred[i] != ensemble_pred[i]):
                div += 1
            elif (y_pred[i] == ensemble_pred[i]):
                div += 0
        return div / len(y_pred)

    def pass_diversity(self, flag_div, val_div, count, pass_error):
        """
        Method that returns if a classifier contributes to the diversity
        """
        threshold_div = 0
        if not flag_div:
            return True, threshold_div
        if not pass_error:
            return True, threshold_div
        if not count != 0:
            return True, threshold_div
        if (len(self.diversities) == 0):
            self.diversities = np.append(self.diversities, val_div)
            return True, threshold_div
        else:
            # d_ens=sum/t_cycles_accepted
            div_ens = np.mean(np.append(self.diversities, val_div))
        # print(self.diversities, val_div, div_ens, self.n_classifiers) # Check
        # behavoir diversity
        threshold_div = self.eta  # self.eta * np.max(self.diversities)
        if div_ens >= threshold_div:
            self.diversities = np.append(self.diversities, val_div)
            return True, threshold_div
        else:
            return False, threshold_div

    def early_stop(self, count, x_test, y_test, gammaVar):
        """
        Method that implements the early stop
        """
        strip_length = 5
        if count == 0 or count % strip_length != 0:
            return False
        test_score = 1 - accuracy_score(y_test, self.predict(x_test))
        if len(self.test_scores) == 0:
            self.test_scores = np.append(self.test_scores, test_score)
            return False
        # Stop if we reached perfect testing score
        min_test_score = np.amin(self.test_scores)
        if (min_test_score == 0):
            print(min_test_score, 'min_test_score')
            return True
        self.test_scores = np.append(self.test_scores, test_score)
        # Early stop definition 3 (see the paper)
        index_test = int(count / strip_length)
        current_error = self.test_scores[index_test - 1]
        past_error = self.test_scores[index_test -
                                      int(strip_length / strip_length) - 1]
        if (current_error == past_error):
            self.count_over_train_equal = np.append(
                self.count_over_train_equal, 1)
        if (current_error > past_error):
            self.count_over_train = np.append(self.count_over_train, 1)
            self.count_over_train_equal = ([])
        if (current_error < past_error):
            self.count_over_train_equal = ([])
        counter_flag = count >= 100
        if (counter_flag):
            counter_flag = count >= 250
        # print('current:', round(current_error,2), ' past:',round(past_error,2), ' count:', count, ' length: ',
        # len(self.count_over_train), 'another check', gammaVar, 'count
        # equal:', len(self.count_over_train_equal))
        return len(self.count_over_train) >= 4 or counter_flag or len(
            self.count_over_train_equal) >= 15  # previous_score <= test_score

    def _check_X_y(self, X, y):
        """"
        Validate assumptions about format of input data. Expecting response variable to be formatted as ±1
        """
        assert set(y) == {-1, 1} or set(y) == {-1} or set(y) == {
            1}  # Extra conditions for highly imbalance
        # If input data already is numpy array, do nothing
        if isinstance(
            X, type(
                np.array(
                    []))) and isinstance(
                y, type(
                    np.array(
                        []))):
            return X, y
        else:
            # Convert pandas into numpy arrays
            X = X.values
            y = y.values
            return X, y

    def decision_thresholds(self, X, glob_dec):
        """
        Method to threshold the svm decision, by varying the bias(intercept)
        We need to calculate the AUC
        """
        svm_decisions = np.array([learner.decision_function(X)
                                 for learner in self.weak_svm])
        svm_biases = np.array(
            [learner.intercept_ for learner in self.weak_svm])
        thres_decision = []
        steps = np.linspace(-10, 10, num=101)
        decision, decision_temp = ([]), ([])
        # Threshold each individual classifier
        if not glob_dec:
            for i in range(len(steps)):
                decision = np.array(
                    [
                        np.sign(
                            svm_decisions[j] -
                            svm_biases[j] +
                            steps[i] *
                            svm_biases[j]) for j in range(
                            len(svm_biases))])
                thres_decision.append(decision)
            thres_decision = np.array(thres_decision)
            final_threshold_decisions = []
            for i in range(len(steps)):
                final = np.sign(np.dot(self.alphas, thres_decision[i]))
                final_threshold_decisions.append(final)
            return np.array(final_threshold_decisions)
        elif glob_dec:  # glob_dec == true threshold the global final classifier
            decision = np.array([svm_decisions[j] + svm_biases[j]
                                for j in range(len(svm_biases))])
            decision = np.dot(self.alphas, decision)
            for i in range(len(steps)):
                # print('check point: ', len(steps), len(decision))
                decision_temp = np.array(
                    [np.sign(decision[j] + steps[i]) for j in range(len(decision))])  # *svm_biases[j]
                thres_decision.append(decision_temp)
            return np.array(thres_decision)

    def number_class(self, X):
        """
        Different number of classifiers
        """
        svm_preds = np.array([learner.predict(X) for learner in self.weak_svm])
        number = []  # Array of predicted samples i.e. array of arrays
        for i in range(len(self.alphas)):
            number.append(self.alphas[i] * svm_preds[i])
        number = np.array(number)
        number = np.cumsum(number, axis=0)
        return np.sign(number)

    def clean(self):
        """
        Method to clean is needed in case of running several times
        when creating only one instance
        """
        self.weak_svm = ([])
        self.alphas = ([])
        self.weights_list = []
        self.errors = ([])
        self.precision = ([])
        self.eta = 0.7
        self.diversities = ([])
        self.Div_total = ([])
        self.Div_partial = ([])
        self.train_scores = ([])
        self.test_scores = ([])
        self.count_over_train = ([])
        self.count_over_train_equal = ([])
        self.n_classifiers = 0
