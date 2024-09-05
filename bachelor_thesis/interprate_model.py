"""
Name: interprate_model.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 20.12.21
Description:
"""

import math
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def interprate_lasso(models: list, X: pd.DataFrame, number: int) -> None:  # noqa: N803, PLR0915
    """

    This function lets you gain inshights into the lasso regression.

    :param models: array : trained models
    :param X: pd.Dataframe : X data
    :param number: interger : number of features to show in the plot
    :return: None
    """
    sns.set_theme(style="ticks")

    # list for storing value for chosen topics
    list_indizes_topics = []
    list_indizes_hyper_parms = []

    # counter
    count_topics = 0

    # check the hyper params
    check_hypers = True

    # if all models are the same:
    if all(x.get_params()["alpha"] == models[0].get_params()["alpha"] for x in models):
        print("All models have identical hyper parameters:")  # noqa: T201
        print(f"{models[0].get_params()}")  # noqa: T201
        print("-" * 10)  # noqa: T201

        # no need to check the alphas anymore
        check_hypers = False

    for model in models:
        # get the fitted coefficients of the trained build_model
        coef = pd.Series(model.coef_, index=X.columns)

        # get the columns that were selected
        columns_to_keep = coef[coef != 0].index

        # get the hyper parameter
        hyper_params = model.get_params()["alpha"]

        # increase counter if model purged all coefficients to zero
        if columns_to_keep.empty:
            # increase counter
            count_topics = count_topics + 1

        else:
            pass

        # append to list topics
        for item in columns_to_keep:
            list_indizes_topics.append(item)  # noqa: PERF402

        # append to list hyper parameters
        list_indizes_hyper_parms.append(hyper_params)

    if count_topics == 242:  # noqa: PLR2004
        print("The coefficients of all models are all 0.")  # noqa: T201

    else:
        if check_hypers is True:
            # get it as counter object
            counting_hyperparams = Counter(list_indizes_hyper_parms)

            # convert to a pandas Dataframe
            df_hyperparams = pd.DataFrame.from_dict(
                counting_hyperparams,
                orient="index",
            )
            df_hyperparams = df_hyperparams.rename(columns={0: "count"})
            df_hyperparams = df_hyperparams.sort_values(["count"], ascending=False)

            # plotting
            sns.set(
                rc={"figure.figsize": (8.6074, 3.17), "figure.dpi": 227},
                style="ticks",
            )
            sns.barplot(
                x=df_hyperparams.index,
                y=df_hyperparams["count"],
                palette="rocket",
            )
            plt.xticks(rotation=45)
            plt.ylabel("Count")
            plt.xlabel("Hyperparameter : Penalization Term")
            plt.gcf().subplots_adjust(bottom=0.25)
            plt.show()

        else:
            pass

        # get it as counter object
        counting_topics = Counter(list_indizes_topics)

        # convert to a pandas Dataframe
        df_topics = pd.DataFrame.from_dict(counting_topics, orient="index")
        df_topics = df_topics.rename(columns={0: "count"})
        df_topics = df_topics.sort_values(["count"], ascending=False)

        # average number of topics
        avg_topics = round(len(list_indizes_topics) / 242, 4)

        print(  # noqa: T201
            f"On average {avg_topics} Topics out of the 180 topics have been selected.",
        )
        print("-" * 10)  # noqa: T201

        # set avg topics as the next higher integer
        avg_topics = math.ceil(avg_topics)

        # if many topics are selected, show only to best
        if avg_topics > 1:
            # only show the most chosen topics
            df_topics = df_topics.head(n=number)

        else:
            pass

        # plotting
        sns.set(rc={"figure.figsize": (8.6074, 3.17), "figure.dpi": 227}, style="ticks")
        sns.barplot(x=df_topics.index, y=df_topics["count"], palette="rocket")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.xlabel("Topic")
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.show()


def feature_importance_rf(models: object, X: pd.DataFrame, number: int = 5) -> None:  # noqa: N803, PLR0915
    """

    This function computes the feature importance in a tree by impurity decrease

    :param models: object : trained model
    :param X: pd.Dataframe : X data
    :param number: interger : number of features to show in the plot
    :return: None
    """
    sns.set_theme(style="ticks")

    # column names
    colnames = X.columns.to_list()

    # list for storing value for chosen hyper parms and feature importances
    list_indizes_hyper_parms_depth = []
    list_indizes_hyper_parms_estimator = []

    # total pandas series to store the feature importance values
    forest_importances = pd.Series(0, index=colnames)

    # check the hyper params
    check_hypers = True

    # if all models are the same:
    if all(str(x) == str(models[0]) for x in models):
        print("All models have identical hyper parameters:")  # noqa: T201
        print(f"{models[0].get_params()}")  # noqa: T201
        print("-" * 10)  # noqa: T201

        # no need to check the hyperparms anymore
        check_hypers = False

    for model in models:
        # get the hyper parameter
        hyper_params_depth = model.get_params()["max_depth"]
        hyper_params_estimators = model.get_params()["n_estimators"]

        # get the feature importance of the models
        forest_importances_per_model = pd.Series(
            model.feature_importances_,
            index=colnames,
        )

        # append to list hyper parameters
        list_indizes_hyper_parms_depth.append(hyper_params_depth)
        list_indizes_hyper_parms_estimator.append(hyper_params_estimators)

        # add the padnas series
        forest_importances = forest_importances + forest_importances_per_model

    if check_hypers is True:
        # get it as counter object
        counting_hyperparams_depth = Counter(list_indizes_hyper_parms_depth)
        counting_hyperparams_estimtor = Counter(list_indizes_hyper_parms_estimator)

        # convert to a pandas Dataframe
        df_hyperparams_depth = pd.DataFrame.from_dict(
            counting_hyperparams_depth,
            orient="index",
        )
        df_hyperparams_depth = df_hyperparams_depth.rename(columns={0: "count"})
        df_hyperparams_depth = df_hyperparams_depth.sort_values(
            ["count"],
            ascending=False,
        )

        df_hyperparams_estimtor = pd.DataFrame.from_dict(
            counting_hyperparams_estimtor,
            orient="index",
        )
        df_hyperparams_estimtor = df_hyperparams_estimtor.rename(columns={0: "count"})
        df_hyperparams_estimtor = df_hyperparams_estimtor.sort_values(
            ["count"],
            ascending=False,
        )

        # plotting
        sns.set(rc={"figure.figsize": (8.6074, 3.17), "figure.dpi": 227}, style="ticks")
        sns.barplot(
            x=df_hyperparams_depth.index,
            y=df_hyperparams_depth["count"],
            palette="rocket",
        )
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.xlabel("Hyperparameter : Depth of Trees")
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.show()

        # plotting
        sns.set(rc={"figure.figsize": (8.6074, 3.17), "figure.dpi": 227}, style="ticks")
        sns.barplot(
            x=df_hyperparams_estimtor.index,
            y=df_hyperparams_estimtor["count"],
            palette="rocket",
        )
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.xlabel("Hyperparameter : Number of Trees")
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.show()

    # divide the forst importances by 242
    forest_importances = forest_importances / 242

    # sort the padnas series accordingly
    forest_importances = forest_importances.sort_values(ascending=False)

    # subset the pandas series
    forest_importances = forest_importances.head(number)

    # plotting
    sns.set(rc={"figure.figsize": (8.6074, 3.17), "figure.dpi": 227}, style="ticks")
    sns.barplot(x=forest_importances.index, y=forest_importances, palette="rocket")
    plt.xticks(rotation=45)
    plt.ylabel("Average Mean Impurity Decrease")
    plt.xlabel("Topic")
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.show()
