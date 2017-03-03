from pylab import *
from common.console import Logger
from common.iteration import piter
from common.classes.Struct import Struct
import sklearn as skl
import scipy
import numpy as np

def compute_negloglikelihood(F, y):
    F_exp = np.exp(F - F[0])
    p = F_exp / np.sum(F_exp, axis=0)
    return -np.sum(np.log(p) * y, axis=0)

def compute_negloglikelihood_deriv(F, y):
    F_exp = np.exp(F - F[0])
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
        use_dropout (boolean): whether to use dropout or not.
        drop_rate (float): probability to drop tree at the time of a single iteration.
        skip_drop (float): probability to skip dropout at the time of a single iteration.
        normalize_type (string): defines weight which assumed to next tree in ensemble.
            "tree" - 1 / (number of dropped trees + 1)
            "forest" - 0.5
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
                 shrinkage=1, use_dropout=False, drop_rate=0.1, skip_drop=0.0, normalize_type='tree', 
                 max_fun_evals=200, xtol=10**-6, ftol=10**-6, log_level=0):
        self.base_learners_count = base_learners_count
        self.base_learner = base_learner
        self.fit_coefs = fit_coefs
        self.shrinkage = shrinkage
        self.log = Logger(log_level)
        self.refit_tree = refit_tree
        self.optimization = Struct(max_fun_evals=max_fun_evals, xtol=xtol, ftol=ftol)

        if use_dropout and (drop_rate != 0) and (skip_drop != 1):
            self.use_dropout = True
            self.drop_rate = drop_rate
            self.skip_drop = skip_drop
            if normalize_type in ['tree', 'forest']:
                self.normalize_type = normalize_type
            else:
                raise Exception('Not implemented normalize_type "%s"' %normalize_type)
        else:
            self.use_dropout = False
        
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
        self.drop_skipped = True  # if True then fit as usually
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
            if self.task == 'regression':
                val_loss = loss
            elif self.classes_count > 2:  # multiclass classification
                Y_val_hat = np.argmax(self.F_val, axis=0)
                val_loss = 1 - skl.metrics.accuracy_score(y_val, Y_val_hat)
            else:
                Y_val_hat = (self.F_val >= 0).astype(int)
                Y_val_hat[Y_val_hat == 0] = -1
                val_loss = 1 - skl.metrics.accuracy_score(y_val, Y_val_hat)
            self.val_err.append(val_loss)

        if loss < self.min_loss:
            self.min_pos = iter_num
            self.min_loss = loss
            self.coefs_min_ = self.coefs[:]

        if iter_num - self.min_pos >= bad_iters_count:
            # save best result
            self.base_learners_count = self.min_pos + 1
            self.coefs = self.coefs_min_
            self.log.pr1(
                '\nEarly stopping with %d base lerners, because last %d losses were above min_loss=%.3f at position %d.' % (
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
        if self.drop_skipped:
            z = -self.loss_derivative(self.F_current, y)
        else:
            # compute loss derivative without dropped base learners
            dropped_inds, retained_inds = self.get_indices(self.drop_rate, iter_num)
            dropped_count = dropped_inds.shape[0]

            F_dropped = zeros(y.shape[0])
            for i in dropped_inds:
                F_dropped += self.coefs[i] * self.base_learners[i].predict(X)
            self.F_current -= F_dropped
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
        if self.drop_skipped:
            self.coefs.append(coef)
            self.base_learners.append(base_learner)
            self.F_current += coef * base_pred
            if self.is_val:
                self.F_val += coef * base_learner.predict(X_val)
        else:
            self.base_learners.append(base_learner)
            if self.normalize_type == 'tree':
                alpha = dropped_count / (dropped_count + 1)
                self.coefs.append(coef / (dropped_count + 1))
            elif self.normalize_type == 'forest':
                alpha = 0.5
                self.coefs.append(coef * 0.5)

            self.F_current += (alpha * F_dropped)
            for i in dropped_inds:
                if self.is_val:
                    self.F_val += (alpha - 1) * self.coefs[i] * self.base_learners[i].predict(X_val)
                self.coefs[i] *= alpha
            self.F_current += self.coefs[-1] * base_pred
            if self.is_val:
                self.F_val += self.coefs[-1] * base_learner.predict(X_val)

        if self.use_dropout:
            self.drop_skipped = (np.random.binomial(1, self.skip_drop) == 1)

        if self.log.level > 0:
            if self.task == 'regression':
                train_loss = mean(self.loss(self.F_current, y))
            else:
                Y_train_hat = (self.F_current >= 0).astype(int)
                Y_train_hat[Y_train_hat == 0] = -1
                train_loss = 1 - skl.metrics.accuracy_score(y, Y_train_hat)
            self.train_err.append(train_loss)
        return
    
    def _make_iter_multiclass(self, X, y, X_val=None, iter_num=None):        
        # compute loss_derivative
        if self.drop_skipped:
            z = -self.loss_derivative(self.F_current, self.y_mask)
        else:
            # compute loss derivative without dropped base learners
            dropped_inds, retained_inds = self.get_indices(self.drop_rate, iter_num)
            dropped_count = dropped_inds.shape[0]
            F_dropped = np.zeros((self.classes_count, y.shape[0]))
            for i in dropped_inds:
                for k in range(self.classes_count):
                    F_dropped[k] += self.coefs[i] * self.base_learners[i][k].predict(X)
            self.F_current -= F_dropped
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
                                         * np.sum(z[k, leaf_pos_sels]) / np.sum(abs_z * (1 - abs_z))
                    base_learner[k].tree_.value[leaf_id, 0, 0] = refined_prediction

        for k in range(self.classes_count):
            base_pred[k] = base_learner[k].predict(X)
        coef = self.shrinkage

        # add new base learners and change coefs
        if self.drop_skipped:
            self.coefs.append(coef)
            self.base_learners.append(base_learner)
            self.F_current += coef * base_pred
            if self.is_val:
                for k in range(self.classes_count):
                    self.F_val[k] += coef * base_learner[k].predict(X_val)
        else:
            self.base_learners.append(base_learner)
            if self.normalize_type == 'tree':
                alpha = dropped_count / (dropped_count + 1)
                self.coefs.append(coef / (dropped_count + 1))
            elif self.normalize_type == 'forest':
                alpha = 0.5
                self.coefs.append(coef * 0.5)

            self.F_current += (alpha * F_dropped)
            for i in dropped_inds:
                if self.is_val:
                    for k in range(self.classes_count):
                        self.F_val[k] += (alpha - 1) * self.coefs[i] * self.base_learners[i][k].predict(X_val)
                self.coefs[i] *= alpha
            self.F_current += self.coefs[-1] * base_pred
            if self.is_val:
                for k in range(self.classes_count):
                    self.F_val[k] += self.coefs[-1] * base_learner[k].predict(X_val)

        if self.use_dropout:
            self.drop_skipped = (np.random.binomial(1, self.skip_drop) == 1)

        if self.log.level > 0:
            Y_train_hat = np.argmax(self.F_current, axis=0)
            train_loss = 1 - skl.metrics.accuracy_score(y, Y_train_hat)
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
        #if self.loss!='log':
        #    raise Exception('Inapliccable for loss %s'%self.loss)
        if self.task != 'classification':
            raise Exception('task must be classification')
        scores = self.F(X, self.base_learners_count)
        if self.classes_count == 2:
            probs = 1 / (1 + exp(-scores))
            return hstack([(1 - probs)[:newaxis], probs[:newaxis]])
        else:
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


    def get_indices(self, drop_rate, base_learners_count):
        '''
        Choice of estimators for dropping.
        drop_rate - probability to drop estimator.
        base_learners_count - number of estimators.

        Returns tuple (dropped_inds, retained_inds), where dropped_inds - numpy array of dropped estimators' indices,
        retained_inds - numpy array of retained estimators' indices.
        '''
        a = np.random.binomial(1, drop_rate, (base_learners_count,))
        if np.all(a == 0):
            a[np.random.randint(0, base_learners_count)] = 1
        dropped_inds = np.where(a == 1)[0]
        retained_inds = np.where(a == 0)[0]
        return (dropped_inds, retained_inds)
