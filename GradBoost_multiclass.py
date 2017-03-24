from pylab import *
from common.console import Logger
from common.iteration import piter
from common.classes.Struct import Struct
import sklearn as skl
import scipy
import numpy as np

def compute_negloglikelihood(F, y):
    F_exp = np.exp(F - np.max(F, axis=0))
    p = F_exp / np.sum(F_exp, axis=0)
    return -np.sum(np.log(p + 10**(-12)) * y, axis=0)

def compute_negloglikelihood_deriv(F, y):
    F_exp = np.exp(F - np.max(F, axis=0))
    p = F_exp / np.sum(F_exp, axis=0)
    return p - y


class GradBoost:
    '''Realization of gradient boosting predictor. 
    Params:
        base_learner (object): instance of base learner class, initialized with necessary parameters. 
        All base learners inside are recreated with these parameters.
        loss (string): loss function. Possible values are: "square", "exp" and "log".
        base_learners_count (int): how many base learners to fit (number of boosting iterations)
        fit_coefs (boolean): whether to fit or not multiplier coefficients by each base learner 
        refit_tree (boolean): in case the base learner is regression tree, whether to refit or not leaf predictions of each tree.
        shrinkage (float): how much to multiply each coefficient by the base learner
        log_level (int): how many debug messages to display. The lower the value, the more logger messages will be shown.
    Comments:
        - Univarite regression predictions are made for loss="square".
        - Binary classification predictions are made for loss="exp" or loss="log". In these cases y=0 or y=1.
        - Multiclass classification predictions are made for loss="negative loglikelihood". In these cases y must be from 0 to classes_number - 1
        - For loss="log" not only classes can be predicted (with predict function) but also class probabilities (with predict_proba function)
        - Zero-th approximation is zero. Higher order approximations are sums of base learners with coefficients.
    Author: 
        Victor Kitov, 03.2016.'''

    def __init__(self, base_learner, base_learners_count, loss=None, fit_coefs=True, refit_tree=True, 
                 shrinkage=1, max_fun_evals=200, xtol=10**-6, ftol=10**-6, log_level=0):
        self.base_learners_count = base_learners_count
        self.base_learner = base_learner
        self.fit_coefs = fit_coefs
        self.shrinkage = shrinkage
        self.log = Logger(log_level)
        self.refit_tree = refit_tree
        self.optimization = Struct(max_fun_evals=max_fun_evals, xtol=xtol, ftol=ftol)
        
        if loss == 'square':
            self.loss = lambda r, y: 0.5 * (r - y)**2
            self.loss_derivative = lambda r, y: (r - y)
            self.task = 'regression'
        elif loss == 'exp':
            self.loss = lambda r, y: exp(-r * y)
            self.loss_derivative = lambda r, y: -(y * exp(-r * y))
            self.task = 'classification'
        elif loss == 'log':
            self.loss = lambda r, y: log(1 + exp(-r * y))
            self.loss_derivative = lambda r, y: -(y / (1 + exp(r * y)))
            self.task = 'classification'
        else:
            raise Exception('Not implemented loss "%s"'%loss)
    
    def fit(self, X, y, X_val=None, y_val=None, bad_iters_count=+inf):
        '''If called like fit(X,y), then the number of base_learners is always base_learners_count (specified at initialization).
        If called like fit(self, X, y, X_val, y_val, bad_iters_count) at most there are also base_learners_count base learners but may be less due to
        early stopping:
        At each iteration accuracy (using validation set, specified by X_val [design matrix], y_val [outputs]) is estimated and
        position of best iteration tracked. If there were >=bad_iters_count after the best iteration, fitting process stops.'''

        X = X.astype(float32)
        y = y.astype(float32)
        self.min_loss = +inf
        self.min_pos = -1
        self.is_val = (X_val is not None) and (y_val is not None)

        self.coefs = []
        self.coefs_min_ = []
        self.base_learners = []

        if self.log.level > 0:
            self.val_err = []
            self.train_err = []

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have equal length.')

        if self.is_val:
            X_val = X_val.astype(float32)
            y_val = y_val.astype(float32)
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError('X_val and y_val must have equal length.')
            if X_val.shape[1] != X.shape[1]:
                raise ValueError('X_val and X must have equal number of features.')

        if self.task == 'classification':
            self._check_class(y, y_val)

        self.F_current = self.F(X) # current value, all zeros if not tuned before
        if self.is_val:
            self.F_val = self.F(X_val)

        for iter_num in piter(range(self.base_learners_count), percent_period=3, show=(self.log.level >= 2)):
            self._make_iter(X, y, X_val, iter_num)
            if self.is_val:
                if self._check_out(y_val, iter_num, bad_iters_count):
                    break
        return

    def _check_class(self, y, y_val=None):
        """Checks the number of classes"""
        unique_values = unique(y)
        self.classes_count = unique_values.shape[0]

        if self.classes_count > 2:
            # multiclass classification with negative loglikelihood loss
            self.loss = compute_negloglikelihood
            self.loss_derivative = compute_negloglikelihood_deriv

            assert all(logical_and(unique_values < self.classes_count, unique_values > -1))
            # y_ki = [y_i = k]
            self.y_mask = zeros((self.classes_count, y.shape[0]))
            for k in range(self.classes_count):
                self.y_mask[k] = (y == k)
        else:
            assert all(unique_values == [0, 1]), 'Only y=0 or y=1 supported!'
            y[y == 0] = -1 # inner format of classes y=+1 or y=-1

        if self.is_val:
            unique_values = unique(y_val)
            if self.classes_count > 2:
                assert all(logical_and(unique_values < self.classes_count, unique_values > -1))
                # y_ki = [y_i = k]
                self.y_val_mask = zeros((self.classes_count, y_val.shape[0]))
                for k in range(self.classes_count):
                    self.y_val_mask[k] = (y_val == k)
            else:
                assert all(unique_values == [0, 1]), 'Only y=0 or y=1 supported!'
                y_val[y_val == 0] = -1

        return

    def _check_out(self, y_val, iter_num, bad_iters_count):
        """The method computes loss on validation set and decides whether to finish learning or not."""

        if (self.task == 'classification') and (self.classes_count > 2):
            loss = np.mean(self.loss(self.F_val, self.y_val_mask))
        else:
            loss = np.mean(self.loss(self.F_val, y_val))

        if self.log.level > 0:
            self.val_err.append(loss)

        if loss < self.min_loss:
            self.min_pos = iter_num
            self.min_loss = loss
            self.coefs_min_ = self.coefs[:]

        if iter_num - self.min_pos >= bad_iters_count:
            # save best result
            self.base_learners_count = self.min_pos + 1
            self.coefs = self.coefs_min_
            self.log.pr1(
                '\nEarly stopping with %d base lerners, because last %d losses were above min_loss=%f at position %d.' % (
                self.base_learners_count,
                bad_iters_count,
                self.min_loss,
                self.min_pos + 1))
            return True
        elif iter_num + 1 == self.base_learners_count:
            self.base_learners_count = self.min_pos + 1
            self.coefs = self.coefs_min_
            return True
        else:
            return False

    def _make_iter(self, X, y, X_val=None, iter_num=None):
        """One iteration of boosting"""
        if (self.task == 'classification') and (self.classes_count > 2):
            return self._make_iter_multiclass(X, y, X_val, iter_num)
        
        # compute loss_derivative
        z = -self.loss_derivative(self.F_current, y)

        # fit new base learners
        base_learner = self.base_learner.__class__(**self.base_learner.get_params()) # recreate base learner
        base_learner.fit(X, z)

        # refitting tree using scipy.optimize
        if isinstance(self.base_learner, skl.tree.tree.DecisionTreeRegressor) and self.refit_tree:
            leaf_ids = base_learner.tree_.apply(X)
            unique_leaf_ids = unique(leaf_ids)
            for leaf_id in unique_leaf_ids:
                leaf_pos_sels = (leaf_ids == leaf_id)
                prediction = base_learner.tree_.value[leaf_id, 0, 0]

                def loss_at_leaf(value):
                    return np.sum(self.loss(self.F_current[leaf_pos_sels] + value, y[leaf_pos_sels]))
                   
                refined_prediction = scipy.optimize.fmin(loss_at_leaf, prediction, xtol=self.optimization.xtol, 
                                                         ftol=self.optimization.ftol, maxfun=self.optimization.max_fun_evals,
                                                         disp=0)
                base_learner.tree_.value[leaf_id, 0, 0] = refined_prediction

        base_pred = base_learner.predict(X)
        if self.fit_coefs == False: # coefficients by base learner refitting
            coef = 1
        else:
            def loss_after_weighted_addition(coef):
                return np.sum(self.loss(self.F_current + coef * base_pred, y))

            res = scipy.optimize.fmin(loss_after_weighted_addition, 1, xtol=self.optimization.xtol,
                                      ftol=self.optimization.ftol, maxfun=self.optimization.max_fun_evals,
                                      disp=0)
            coef = res[0]
            if coef < 0:
                self.log.pr3('coef=%s is negative!' % coef)
            if coef == 0:
                self.log.pr3('coef=%s is zero!' % coef)
        coef *= self.shrinkage

        # add new base learners and change coefs
        self.coefs.append(coef)
        self.base_learners.append(base_learner)
        self.F_current += coef * base_pred
        if self.is_val:
            self.F_val += coef * base_learner.predict(X_val)

        if self.log.level > 0:
            train_loss = np.mean(self.loss(self.F_current, y))
            self.train_err.append(train_loss)

        return
    
    def _make_iter_multiclass(self, X, y, X_val=None, iter_num=None):        
        # compute loss_derivative
        z = -self.loss_derivative(self.F_current, self.y_mask)

        # fit new base learners
        base_learner = []
        base_pred = zeros((self.classes_count, y.shape[0]))
        for k in range(self.classes_count):
            base_learner.append(self.base_learner.__class__(**self.base_learner.get_params()))
            base_learner[k].fit(X, z[k])

        # refitting according to Friedman
        if isinstance(self.base_learner, skl.tree.tree.DecisionTreeRegressor) and self.refit_tree:
            for k in range(self.classes_count):
                leaf_ids = base_learner[k].tree_.apply(X)
                unique_leaf_ids = unique(leaf_ids)
                for leaf_id in unique_leaf_ids:
                    leaf_pos_sels = (leaf_ids == leaf_id)
                    abs_z = np.abs(z[k, leaf_pos_sels])
                    refined_prediction = (self.classes_count - 1) / self.classes_count \
                                         * np.sum(z[k, leaf_pos_sels]) / (np.sum(abs_z * (1 - abs_z)) + 10**(-9))
                    base_learner[k].tree_.value[leaf_id, 0, 0] = refined_prediction

        for k in range(self.classes_count):
            base_pred[k] = base_learner[k].predict(X)
        coef = self.shrinkage

        # add new base learners and change coefs
        self.coefs.append(coef)
        self.base_learners.append(base_learner)
        self.F_current += coef * base_pred
        if self.is_val:
            for k in range(self.classes_count):
                self.F_val[k] += coef * base_learner[k].predict(X_val)

        if self.log.level > 0:
            train_loss = np.mean(self.loss(self.F_current, self.y_mask))
            self.train_err.append(train_loss)

        return        

    def F(self, X, max_base_learners_count=inf):
        '''Internal function used for forecasting. 
           X-design matrix, each row is an object for which a forecast should be made. 
           max_base_learners_count - maximal iteration at which to stop. F is evaluated for min(max_base_learners_count, len(self.base_learners)) models.'''

        if (self.task == 'classification') and (self.classes_count > 2):
            F_val = zeros((self.classes_count, X.shape[0]))
            for iter_num, (coef, base_learner) in enumerate(zip(self.coefs, self.base_learners)):
                for k in range(self.classes_count):
                    F_val[k] += coef * base_learner[k].predict(X)
                if iter_num + 1 >= max_base_learners_count:
                    break
        else:
            F_val = zeros(X.shape[0])
            for iter_num, (coef, base_learner) in enumerate(zip(self.coefs, self.base_learners)):
                F_val += coef * base_learner.predict(X)
                if iter_num + 1 >= max_base_learners_count:
                    break
        return F_val
    
    def predict(self, X, base_learners_count=inf):
        if self.task == 'regression':
            return self.F(X, self.base_learners_count)
        elif self.classes_count == 2:  # binary classification
            return (self.F(X, self.base_learners_count) >= 0).astype(int)
        else: # multiclass classification
            return np.argmax(self.F(X, self.base_learners_count), axis=0)
   
    def predict_proba(self, X, base_learners_count=inf):
        '''Predict class probabilities for objects, specified by rows of matrix X. 
        iter_num - at what iteration to stop. If not specified all base learners are used.
        Applicable only for loss function="log". Classes are stored in self.classes_ attribute.'''

        if self.task != 'classification':
            raise Exception('task must be classification')
        scores = self.F(X, self.base_learners_count)
        if self.classes_count == 2: # binary
            probs = 1 / (1 + exp(-scores))
            return hstack([(1 - probs)[:newaxis], probs[:newaxis]])
        else: # multiclass
            scores_exp = np.exp(self.F_current)
            probs = scores_exp / np.sum(scores_exp, axis=0)
            return probs.transpose()
    
    def get_losses(self, X, Y):
        '''Estimate loss for each iteration of the prediction process of boosting.
        Returns an array losses with length=#{base learners}'''
        
        losses = zeros(len(self.coefs))
        F_val = zeros(len(X))

        for iter_num, (coef, base_learner) in enumerate(zip(self.coefs, self.base_learners)):
            F_val += coef * base_learner.predict(X)

            if self.task == 'regression':
                Y_hat = F_val
                losses[iter_num] = mean(abs(Y_hat-Y))
            else:  # classification
                Y_hat = (F_val >= 0).astype(int)
                losses[iter_num] = 1 - skl.metrics.accuracy_score(Y, Y_hat)

        return losses

    def remove_redundant_base_learners(self, X_val, Y_val, max_count=+inf):
        '''
        Using validation set, specified by (X_val, y_val) find optimal number of base learners
        (at this number the loss on validation is minimal).
        All base learners above this number are removed. 
        max_count - is the maximum possible number of base learners retained.
        '''
        orig_count = len(self.base_learners)
        losses = self.get_losses(X_val, Y_val)
        self.base_learners_count = min(argmin(losses)+1, max_count)
        self.coefs = self.coefs[:self.base_learners_count]
        self.base_learners = self.base_learners[:self.base_learners_count]
        self.log.pr1('Cut base learners count from %d to %d.' %(orig_count, self.base_learners_count))

