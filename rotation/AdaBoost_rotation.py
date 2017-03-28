import numpy as np

from sklearn.base import clone, BaseEstimator

from .misc import pca_rotation
from .misc import IndexGenerator

MAX_INT = np.iinfo(np.int32).max


class RotationAdaBoost(BaseEstimator):
    def __init__(self,
                 base_learner,
                 base_learners_count,
                 max_features_in_subset='log',
                 enable_weighted_rotation=False,
                 samples_fraction=0.75,
                 rotation_func=pca_rotation,
                 shrinkage=0.1,
                 verbose=0,
                 random_state=None):

        self.base_learner = base_learner
        self.base_learners_count = base_learners_count
        self.shrinkage = shrinkage

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

        self.enable_weighted_rotation = enable_weighted_rotation,
        self.samples_fraction = samples_fraction
        self.rotation_func = rotation_func
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.n_samples_, self.n_features_ = X.shape

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Check parameters
        if self.shrinkage <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if self.max_features_in_subset(self.n_features_) > self.n_features_:
            raise ValueError("max_features_in_subset=%d must be smaller or equal than"
                             " n_features=%d"
                             % (self.max_features_in_subset(self.n_features_), self.n_features_))

        random_state = np.random.RandomState(seed=self.random_state)

        sample_weight = np.empty(self.n_samples_, dtype=np.float64)
        sample_weight[:] = 1. / self.n_samples_

        self.estimators_ = []
        self.rot_matrices_ = []
        self.estimator_weights_ = np.zeros(self.base_learners_count, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.base_learners_count, dtype=np.float64)

        for it in range(self.base_learners_count):

            sample_weight, estimator_weight, estimator_error = \
                self._boost_iteration(
                    it,
                    X, y,
                    sample_weight,
                    random_state.randint(MAX_INT)
                )

            self.estimator_weights_[it] = estimator_weight
            self.estimator_errors_[it] = estimator_error

            sample_weight_sum = np.sum(sample_weight)

            if it < self.base_learners_count - 1:
                sample_weight /= sample_weight_sum

        return self

    def _make_rotation_matrix(self, X, sample_weight, random_state):
        rot_matrix = np.zeros((self.n_features_, self.n_features_), dtype=np.float32)

        index_gen = IndexGenerator((self.n_samples_, self.n_features_),
                                   self.max_features_in_subset(self.n_features_),
                                   self.samples_fraction,
                                   random_state)

        for row_inds, col_inds in index_gen:
            Xi = X[row_inds[:, None], col_inds[None, :]]

            if self.enable_weighted_rotation:
                Xi *= np.sqrt(sample_weight[row_inds, None])

            Xi_components = self.rotation_func(Xi)

            rot_matrix[col_inds[:, None], col_inds[None, :]] = Xi_components

        return rot_matrix

    def _boost_iteration(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        rot_matrix = self._make_rotation_matrix(X, sample_weight, random_state)

        estimator = clone(self.base_learner)
        estimator.set_params(random_state=random_state)
        estimator.fit(X.dot(rot_matrix), y, sample_weight=sample_weight)

        self.rot_matrices_.append(rot_matrix)
        self.estimators_.append(estimator)

        y_predict = estimator.predict(X.dot(rot_matrix))

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_predict != y

        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:

            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        estimator_weight = self.shrinkage * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        if not iboost == self.base_learners_count - 1:
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        pred = sum((estimator.predict(X.dot(R)) == classes).T * w
                   for estimator, w, R in zip(self.estimators_,
                                              self.estimator_weights_,
                                              self.rot_matrices_))

        pred /= self.estimator_weights_.sum()

        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)

        return pred

    def predict_proba(self, X):
        n_classes = self.n_classes_

        proba = sum(estimator.predict_proba(X.dot(R)) * w
                    for estimator, w, R in zip(self.estimators_,
                                               self.estimator_weights_,
                                               self.rot_matrices_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba

    def staged_predict(self, X):
        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(
                    np.argmax(pred, axis=1), axis=0))

    def staged_decision_function(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator, rot_matrix in zip(self.estimator_weights_,
                                                 self.estimators_,
                                                 self.rot_matrices_):
            norm += weight

            current_pred = estimator.predict(X.dot(rot_matrix))
            current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    def staged_predict_proba(self, X):
        n_classes = self.n_classes_
        proba = None
        norm = 0.

        for weight, estimator, rot_matrix in zip(self.estimator_weights_,
                                                 self.estimators_,
                                                 self.rot_matrices_):
            norm += weight

            current_proba = estimator.predict_proba(X.dot(rot_matrix)) * weight

            if proba is None:
                proba = current_proba
            else:
                proba += current_proba

            real_proba = np.exp((1. / (n_classes - 1)) * (proba / norm))
            normalizer = real_proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            real_proba /= normalizer

            yield real_proba
