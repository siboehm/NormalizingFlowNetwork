import os
import numpy as np
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import pandas
import tensorflow as tf
from tensorflow.python import tf2
from cde.density_simulation import SkewNormal, GaussianMixture, ArmaJump, EconDensity

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

DENSITY_CLASSES = {
    "SkewNormal": SkewNormal,
    "ArmaJump": ArmaJump,
    "EconDensity": EconDensity,
    "GaussianMixture": GaussianMixture,
}

TEST_SIZE = 5 * 10 ** 5


def run_configuation(
    estimator_list, density_list, n_epochs, n_folds, n_datapoints_list, results_dir
):
    for estimator in estimator_list:
        for density_name, density_params in density_list:
            for n_datapoints in n_datapoints_list:
                x_train, y_train = DENSITY_CLASSES[density_name](**density_params).simulate(
                    n_samples=n_datapoints + TEST_SIZE
                )
                # Arma Jump has a dimension different from the others
                if density_name == "ArmaJump":
                    x_train = np.expand_dims(x_train, axis=1)
                    y_train = np.expand_dims(y_train, axis=1)

                estimator["param_grid"]["n_dims"] = [y_train.shape[1]]
                if estimator["estimator_name"] == "bayesian":
                    estimator["param_grid"]["kl_weight_scale"] = [0.5 / n_datapoints]

                model = tf.keras.wrappers.scikit_learn.KerasRegressor(
                    build_fn=estimator["build_fn"], epochs=n_epochs, verbose=0
                )

                cv = GridSearchCV(
                    estimator=model,
                    param_grid=estimator["param_grid"],
                    scoring=estimator["scoring_fn"],
                    n_jobs=-1,
                    pre_dispatch="1*n_jobs",
                    iid=True,
                    cv=ShuffleSplit(n_splits=n_folds, train_size=n_datapoints, random_state=22),
                    refit=False,
                    verbose=10,
                    error_score=np.NaN,
                    return_train_score=False,
                )

                cv.fit(x_train, y_train)

                df = save_single_result(
                    cv.cv_results_, density_name, estimator, n_datapoints, results_dir
                )
                append_to_full_result(df, results_dir)


def append_to_full_result(df, results_dir):
    results_filename = os.path.join(results_dir, "results.csv")
    if not os.path.isfile(results_filename):
        df.to_csv(results_filename)
    else:
        results_df = pandas.read_csv(results_filename)
        results_df = results_df.append(df, ignore_index=True)
        results_df.to_csv(results_filename)


def save_single_result(result_dict, density_name, estimator, n_datapoints, results_dir):
    df = pandas.DataFrame(result_dict)
    df = df.join(pandas.Series([estimator["estimator_name"]] * len(df), name="estimator"))
    df = df.join(pandas.Series([density_name] * len(df), name="density"))
    df = df.join(pandas.Series([n_datapoints] * len(df), name="n_datapoints"))
    filename = "results_{}_{}_{}.csv".format(
        estimator["estimator_name"], density_name, n_datapoints
    )
    df.to_csv(os.path.join(results_dir, filename))
    return df
