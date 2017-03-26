import numpy as np


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
