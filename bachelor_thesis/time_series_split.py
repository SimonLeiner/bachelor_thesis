"""
Name: time_series_split.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 14.12.21
Description: from sklearn Time series split Class adjusted in a sense of book advances in financial machine learning

See: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/model_selection/_split.py#L944
"""

from typing import Optional

import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class TimeSeriesSplitAdjusted(KFold):
    """
    Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Be aware that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <time_series_split>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    max_train_size : int, default=None
        Maximum size for a single training set.

    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 1)``, which is the maximum allowed value
        with ``gap=0``.

        .. versionadded:: 0.24

    gap : int, default=0
        Number of samples to exclude from the end of each train set before
        the test set.

        .. versionadded:: 0.24

    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i`` th split,
    with a test set of size ``n_samples//(n_splits + 1)`` by default,
    where ``n_samples`` is the number of samples.
    """

    def __init__(  # noqa: PLR0913
        self,
        n_splits: int = 5,
        *,
        rolling_size: int = 12,
        min_train_size: int = 120,
        min_test_size: int = 12,
        pct_purge: int = 0.02,
        estimation_horizon: Optional[int] = None,  # noqa: FA100
    ) -> None:
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size
        self.rolling_size = rolling_size
        self.pct_purge = pct_purge
        self.estimation_horizon = estimation_horizon

    def split(self, X: list, y: Optional[list] = None, groups: Optional[list] = None):  # noqa: ANN201, FA100, N803
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)  # noqa: N806
        n_samples = _num_samples(X)
        gap = int(n_samples * self.pct_purge)
        indices = np.arange(n_samples)

        test_starts = range(self.min_train_size + gap, n_samples, self.rolling_size)

        print(  # noqa: T201
            f"Rolling window estimation with : Windowsize {self.rolling_size} | Minimum {self.min_train_size} Training | GAP {gap}",
        )

        # for each sptlit do:
        for test_start in test_starts:
            # calculate the ending index of the training dataset
            train_end = test_start - gap

            # index of the last testing datapoint
            last_train_index = test_start + self.estimation_horizon

            # slicing eg 160 values returns 0 - 159 (160 values)
            print(  # noqa: T201
                f"Training Starts : {indices[0]} - Ended : {indices[train_end]-1} - [GAP:{gap}] - Testing : {indices[test_start:last_train_index]}",
            )
            yield (indices[:train_end], indices[test_start:last_train_index])
