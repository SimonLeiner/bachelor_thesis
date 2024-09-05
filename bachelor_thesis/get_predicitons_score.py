"""
Name: get_predicitons_score.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 15.12.21
Description: Estimation procedure overwritten and adapted from sklearn
"""

import numbers
import time

import numpy as np
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.metrics._scorer import check_scoring
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.model_selection._validation import _check_fit_params
from sklearn.model_selection._validation import _insert_error_scores
from sklearn.model_selection._validation import _normalize_score_results
from sklearn.model_selection._validation import _num_samples
from sklearn.model_selection._validation import _safe_split
from sklearn.model_selection._validation import _score
from sklearn.model_selection._validation import _warn_about_fit_failures
from sklearn.model_selection._validation import format_exc
from sklearn.utils import indexable
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed
from sklearn.utils._joblib import logger


def _fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    param_grid=None,
    tune=False,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    return_predictions=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
) -> dict:
    """Fit estimator and compute scores for a given dataset split.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_score : bool, default=False
        Compute and return score on training set.
    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.
    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).
    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).
    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.
    return_times : bool, default=False
        Whether to return the fit/score times.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        msg = (
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )
        raise ValueError(
            msg,
        )

    progress_msg = ""
    if verbose > 2:  # noqa: PLR2004
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:  # noqa: PLR2004
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:  # noqa: PLR2004
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")  # noqa: T201

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)  # noqa: N806
    X_test, y_test = _safe_split(estimator, X, y, test, train)  # noqa: N806

    # hyper parameter tuning
    if tune is True:
        # grid search the hyper parameters
        param_grid = param_grid if param_grid is not None else {}
        gs = GridSearchCV(
            estimator,
            param_grid=param_grid,
            cv=TimeSeriesSplit(3),
            scoring=scorer,
            n_jobs=1,
        )
        gs.fit(X_train, y_train)
        parameters = gs.best_params_

    else:
        # the defined parameters
        parameters = param_grid

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        if isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer, error_score)

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:  # noqa: PLR2004
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)  # noqa: T201

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    if return_predictions:
        result["predictions"] = estimator.predict(X_test)

    return result


def cross_validate_adjusted(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    param_grid=None,
    tune=False,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    return_predictions=False,
    error_score=np.nan,
) -> dict:
    """Evaluate metric(s) by cross-validation and also record fit/score times.
    Read more in the :ref:`User Guide <multimetric_cross_validation>`.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
        If `scoring` represents a single score, one can use:
        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.
        If `scoring` represents multiple scores, one can use:
        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.
        See :ref:`multimetric_grid_search` for an example.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`.Fold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.
        .. versionadded:: 0.19
        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``
    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.
        .. versionadded:: 0.20
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
        .. versionadded:: 0.20

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:
            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split.
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.
    """
    X, y, groups = indexable(X, y, groups)  # noqa: N806

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator),
            X,
            y,
            scorers,
            train,
            test,
            verbose,
            None,
            fit_params,
            param_grid=param_grid,
            tune=tune,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            return_predictions=return_predictions,
            error_score=error_score,
        )
        for train, test in cv.split(X, y, groups)
    )

    _warn_about_fit_failures(results, error_score)

    # For callabe scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]

    if return_estimator:
        ret["estimator"] = results["estimator"]

    if return_predictions:
        ret["predictions"] = results["predictions"]

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret[f"test_{name}"] = test_scores_dict[name]
        if return_train_score:
            key = f"train_{name}"
            ret[key] = train_scores_dict[name]

    return ret
