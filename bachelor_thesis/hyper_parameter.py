"""
Name: hyper_parameter.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 11.01.22
Description: hyper parameters for the differnetn ML models
"""


def get_hyper_tuning(model: object) -> dict:
    """

    This function return the hyper parameter grid to passed to the gridsearch algorithm.

    :param model: object : build_model to hypertune
    :return: dict : paramters to experiement with
    """
    # get the build_model name
    modelname = type(model).__name__

    # select a Parameter Grid for your build_model:

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    if modelname == "Lasso":
        param_grid = {
            "alpha": [
                0.000003,
                0.000005,
                0.000006,
                0.000007,
                0.00001,
                0.00002,
                0.00003,
                0.00004,
                0.00005,
                0.0001,
                0.001,
                0.002,
                0.003,
                0.005,
                0.007,
                0.05,
                0.5,
                1,
            ],
        }

    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    elif modelname == "RandomForestRegressor":
        param_grid = {
            "n_estimators": [1, 2, 25, 50, 100, 500],
            "max_depth": [1, 2, 3, 5, 10, 100],
            "n_jobs": [-1],
        }

    else:
        print(  # noqa: T201
            f"Couldn't tune the Hyper parameters as the Model with name {modelname} couldn't match.",
        )
        param_grid = {}

    return param_grid
