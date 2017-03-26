from common.console import Logger
from common.iteration import piter
from common.classes.Struct import Struct

import numpy as np

import scipy
import sklearn

MAX_INT = np.iinfo(np.int32).max


def pca_rotation(X):
    """
    Extract principle directions based on a matrix X

    Parameters
    ----------
    X: Array of shape (N, D)
        A design matrix

    Returns
    -------
    V: Array of shape (D, D)
        A matrix of principle components (right-singular vectors of X) ordered by rows

    """
    _, _, V = np.linalg.svd(X - np.mean(X, axis=0), full_matrices=False)

    return V.T


class IndexGenerator(object):
    def __init__(self, shape, max_features_in_subset, samples_fraction, random_state=None):
        self.n_samples, self.n_features = shape
        self.max_features_in_subset = max_features_in_subset
        self.samples_fraction = samples_fraction
        self.random_state = random_state

        self.n_subsets_ = self.n_features // self.max_features_in_subset
        self.last_subset_size_ = self.n_features % self.max_features_in_subset
        self.n_subsets_ += 1 if self.last_subset_size_ != 0 else 0

    def __get_cols(self):
        rnd = np.random.RandomState(seed=self.random_state)

        subset_sizes = np.empty(self.n_subsets_, dtype=np.int32)
        subset_sizes[:] = self.max_features_in_subset

        if self.last_subset_size_ != 0:
            subset_sizes[-1] = self.last_subset_size_

        partition = np.repeat(np.arange(self.n_subsets_, dtype=np.int32), subset_sizes)
        partition = rnd.permutation(partition)

        for i in range(self.n_subsets_):
            yield np.where(partition == i)[0]

    def __get_rows(self):
        rnd = np.random.RandomState(seed=self.random_state)

        for i in range(self.n_subsets_):
            mask = rnd.choice([True, False],
                              size=self.n_samples,
                              p=[self.samples_fraction, 1 - self.samples_fraction])

            yield np.where(mask)[0]

    def __iter__(self):
        return zip(self.__get_rows(), self.__get_cols())


class GradBoost:
    """
    Realization of rotation gradient boosting predictor.

    Params:
        - base_learner (object): instance of base learner class, initialized with necessary
        parameters.
        All base learners inside are recreated with these parameters.
        - loss (string): loss function. Possible values are: "square", "exp" and "log".
        base_learners_count (int): how many base learners to fit (number of boosting iterations)
        - fit_coefs (boolean): whether to fit or not multiplier coefficients by each base learner
        - refit_tree (boolean): in case the base learner is regression tree, whether to refit or
         not leaf predictions of each tree.
        - shrinkage (float): how much to multiply each coefficient by the base learner
        - log_level (int): how many debug messages to display. The lower the value, the more
        logger messages will be shown.

    Extra Params:
        - max_features_in_subset (int): A maximum size of random feature subsets
        - samples_fraction (float): A fraction of  samples to draw for each classifier
        - rotation_func (callable): Function-extractor, applied to the design matrix on each step

    Comments:
        - Univarite regression predictions are made for loss="square".
        - Binary classification predictions are made for loss="exp" or loss="log". In these cases
        y=0 or y=1.
        - For loss="log" not only classes can be predicted (with predict function) but also class
        probabilities (with predict_proba function)
        - Zero-th approximation is zero. Higher order approximations are sums of base learners
        with coefficients.

    Author:
        Victor Kitov, 03.2016.
    """

    def __init__(self, base_learner, base_learners_count, loss=None, fit_coefs=True,
                 refit_tree=True, shrinkage=1, max_fun_evals=200, xtol=10 ** -6,
                 ftol=10 ** -6, log_level=0,
                 max_features_in_subset='log',
                 samples_fraction=0.75,
                 rotation_func=pca_rotation,
                 enable_weighted_rotation=False,
                 random_state=None):
        self.base_learners_count = base_learners_count
        self.base_learner = base_learner
        self.fit_coefs = fit_coefs
        self.shrinkage = shrinkage
        self.log = Logger(log_level)
        self.refit_tree = refit_tree
        self.optimization = Struct(max_fun_evals=max_fun_evals, xtol=xtol, ftol=ftol)

        if max_features_in_subset == 'log':
            self.max_features_in_subset = lambda x: int( np.log2(x) )
        elif max_features_in_subset == 'sqrt':
            self.max_features_in_subset = lambda x: int( np.sqrt(x) )
        elif max_features_in_subset == 'identity':
            self.max_features_in_subset = lambda x: x
        elif max_features_in_subset == 'div2':
            self.max_features_in_subset = lambda x: int( x / 2 )
        elif isinstance(max_features_in_subset, int):
            self.max_features_in_subset = lambda x: max_features_in_subset
        else:
            raise ValueError( 'Unknown function: {}'.format(max_features_in_subset) )

        self.samples_fraction = samples_fraction
        self.rotation_func = rotation_func
        self.enable_weighted_rotation = enable_weighted_rotation
        self.random_state=random_state

        self.coefs = []
        self.base_learners = []
        self.rot_matrices = []

        if loss == 'square':
            self.loss = lambda r, y: 0.5 * (r - y) ** 2
            self.loss_derivative = lambda r, y: (r - y)
            self.task = 'regression'
        elif loss == 'exp':
            self.loss = lambda r, y: np.exp(-r * y)
            self.loss_derivative = lambda r, y: -(y * np.exp(-r * y))
            self.task = 'classification'
        elif loss == 'log':
            self.loss = lambda r, y: np.log(1 + np.exp(-r * y))
            self.loss_derivative = lambda r, y: -(y / (1 + np.exp(r * y)))
            self.task = 'classification'
        else:
            raise Exception('Not implemented loss "%s"' % loss)

    def fit(self, X, y, X_val=None, y_val=None, bad_iters_count=+np.inf):
        """
        If called like fit(X,y), then the number of base_learners is always base_learners_count (specified at initialization).
        If called like fit(self, X, y, X_val, y_val, bad_iters_count) at most there are also base_learners_count base learners but may be less due to
        early stopping:
        At each iteration accuracy (using validation set, specified by X_val [design matrix], y_val [outputs]) is estimated and
        position of best iteration tracked. If there were >=bad_iters_count after the best iteration, fitting process stops.
        """
        random_state = np.random.RandomState(seed=self.random_state)

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        min_loss = +np.inf
        min_iter = -1

        N, D = X.shape

        if self.task == 'classification':
            assert all(np.unique(y) == [0, 1]), 'Only y=0 or y=1 supported!'
            y[y == 0] = -1  # inner format of classes y=+1 or y=-1

        F_current = self.F(X)  # current value, all zeros if not tuned before.

        if X_val is not None and y_val is not None:
            F_val = self.F(X_val)

        for iter_num in piter(range(self.base_learners_count), percent_period=3, show=(self.log.level >= 2)):

            if X_val is not None and y_val is not None:

                if self.task == 'regression':
                    y_val_hat = F_val
                    loss = sklearn.metrics.mean_absolute_error(y_val, y_val_hat)
                elif self.task == 'classification':
                    y_val_hat = np.int32(F_val >= 0)
                    loss = 1 - sklearn.metrics.accuracy_score(y_val, y_val_hat)

                if loss < min_loss:
                    min_iter = iter_num
                    min_loss = loss

                if iter_num - min_iter >= bad_iters_count:
                    self.base_learners_count = len(self.base_learners)

                    msg = '\nEarly stopping with %d base lerners, because last %d losses were above min_loss=%.3f at position %d.'

                    self.log.pr1(
                        msg % (
                        self.base_learners_count,
                        bad_iters_count,
                        min_loss,
                        min_iter))

                    break

            z = -self.loss_derivative(F_current, y)

            base_learner = self.base_learner.__class__(**self.base_learner.get_params())  # recreate base learner

            rot_matrix = np.zeros((D, D), dtype=np.float32)

            if self.enable_weighted_rotation:
                weights = np.abs(z)
                weights /= np.sum(weights)

            index_gen = IndexGenerator(X.shape,
                                       self.max_features_in_subset(D),
                                       self.samples_fraction,
                                       random_state.randint(MAX_INT))

            for row_inds, col_inds in index_gen:
                Xi = X[row_inds[:, None], col_inds[None, :]]

                if self.enable_weighted_rotation:
                    Xi *= np.sqrt(weights[row_inds, None])

                Xi_components = self.rotation_func(Xi)

                di = col_inds.shape[0]
                di_hat = Xi_components.shape[1]

                Xi_components = np.pad(Xi_components,
                                       pad_width=((0, 0), (0, di - di_hat)),
                                       mode='constant')

                rot_matrix[col_inds[:, None], col_inds[None, :]] = Xi_components

            XR = X.dot(rot_matrix)
            base_learner.fit(XR, z)

            if isinstance(base_learner, sklearn.tree.DecisionTreeRegressor) and \
                    (self.refit_tree == True):  # tree refitting
                leaf_ids = base_learner.tree_.apply(XR)
                unique_leaf_ids = np.unique(leaf_ids)

                for leaf_id in unique_leaf_ids:
                    leaf_pos_sels = (leaf_ids == leaf_id)
                    prediction = base_learner.tree_.value[leaf_id, 0, 0]

                    def loss_at_leaf(value):
                        return np.sum(self.loss(F_current[leaf_pos_sels] + value, y[leaf_pos_sels]))

                    refined_prediction = scipy.optimize.fmin(
                        loss_at_leaf,
                        prediction,
                        xtol=self.optimization.xtol,
                        ftol=self.optimization.ftol,
                        maxfun=self.optimization.max_fun_evals,
                        disp=0
                    )

                    base_learner.tree_.value[leaf_id, 0, 0] = refined_prediction

            base_pred = base_learner.predict(XR)

            if not self.fit_coefs:  # coefficients by base learner refitting
                coef = 1
            else:
                def loss_after_weighted_addition(coef):
                    return np.sum(self.loss(F_current + coef * base_pred, y))

                res = scipy.optimize.fmin(loss_after_weighted_addition, 1,
                                          xtol=self.optimization.xtol,
                                          ftol=self.optimization.ftol,
                                          maxfun=self.optimization.max_fun_evals,
                                          disp=0)
                coef = res[0]

                if coef < 0:
                    self.log.pr3('coef=%s is negative!' % coef)
                if coef == 0:
                    self.log.pr3('coef=%s is zero!' % coef)

            coef *= self.shrinkage

            self.coefs.append(coef)
            self.base_learners.append(base_learner)
            self.rot_matrices.append(rot_matrix)

            F_current += coef * base_pred
            if X_val is not None and y_val is not None:
                F_val += coef * base_learner.predict(X_val.dot(rot_matrix))

    def F(self, X, max_base_learners_count=np.inf):
        """
        Internal function used for forecasting.
        X-design matrix, each row is an object for which a forecast should be made.
        max_base_learners_count - maximal iteration at which to stop. F is evaluated for min(max_base_learners_count, len(self.base_learners)) models.
        """

        F_val = np.zeros(len(X))

        for iter_num, (coef, base_learner, rot_matrix) in enumerate(zip(self.coefs, self.base_learners, self.rot_matrices)):
            XR = X.dot(rot_matrix)

            base_pred = base_learner.predict(XR)
            F_val += coef * base_pred

            if iter_num + 1 >= max_base_learners_count:
                break

        return F_val

    def staged_F(self, X, max_base_learners_count=np.inf):
        """
        Function-generator using to compute a value of F for all possible ensemble sizes:
        from 1 to base_learners_count

        """
        F_val = np.zeros(len(X))

        for iter_num, (coef, base_learner, rot_matrix) in enumerate(
                zip(self.coefs, self.base_learners, self.rot_matrices)):
            XR = X.dot(rot_matrix)

            base_pred = base_learner.predict(XR)
            F_val += coef * base_pred

            yield F_val

            if iter_num + 1 >= max_base_learners_count:
                break

    def predict(self, X, base_learners_count=np.inf):
        if self.task == 'regression':
            return self.F(X, base_learners_count)
        else:  # classification
            return np.int32(self.F(X, base_learners_count) >= 0)  # F(X)>=0 = > predition=1 otherwise prediction=0

    def staged_predict(self, X, base_learners_count=np.inf):

        for F_val in self.staged_F(X, base_learners_count):

            if self.task == 'regression':
                yield F_val
            else:  # classification
                yield np.int32(F_val >= 0)

    def predict_proba(self, X, base_learners_count=np.inf):
        """
        Predict class probabilities for objects, specified by rows of matrix X.
        iter_num - at what iteration to stop. If not specified all base learners are used.
        Applicable only for loss function="log". Classes are stored in self.classes_ attribute.
        """

        if self.loss != 'log':
            raise Exception('Inapliccable for loss %s' % self.loss)

        self.classes_ = [0, 1]
        scores = self.F(X, base_learners_count)
        probs = 1 / (1 + np.exp(-scores))
        return np.hstack([1 - probs, probs])

    def get_losses(self, X, Y):
        """
        Estimate loss for each iteration of the prediction process of boosting.
        Returns an array losses with length=#{base learners}
        """

        losses = np.zeros(len(self.base_learners))
        F_val = np.zeros(len(X))

        for iter_num, (coef, base_learner, rot_matrix) in enumerate(zip(self.coefs, self.base_learners, self.rot_matrices)):
            XR = X.dot(rot_matrix)

            base_pred = base_learner.predict(XR)
            F_val += coef * base_pred

            if self.task == 'regression':
                Y_hat = F_val
                losses[iter_num] = (np.mean(abs(Y_hat - Y)) / np.mean(abs(Y)))
            else:  # classification
                Y_hat = np.int32(F_val >= 0)
                losses[iter_num] = 1 - sklearn.metrics.accuracy_score(Y, Y_hat)

        return losses

    def remove_redundant_base_learners(self, X_val, Y_val, max_count=+np.inf):
        """
        Using validation set, specified by (X_val, y_val) find optimal number of base learners
        (at this number the loss on validation is minimal).
        All base learners above this number are removed.

        max_count - is the maximum possible number of base learners retained.
        """

        orig_count = len(self.base_learners)
        losses = self.get_losses(X_val, Y_val)

        self.base_learners_count = min(np.argmin(losses) + 1, max_count)
        self.coefs = self.coefs[:self.base_learners_count]
        self.base_learners = self.base_learners[:self.base_learners_count]
        self.log.pr1('Cut base learners count from %d to %d.' % (orig_count, self.base_learners_count))
