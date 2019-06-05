import os
import numpy as np
from sklearn.model_selection import GridSearchCV
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


def run_configuation(
    estimator_list, density_list, n_epochs, n_folds, n_datapoints, results_dir
):
    for estimator in estimator_list:
        for density_name, density_params in density_list:
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
                cv=n_folds,
                refit=False,
                verbose=10,
                error_score=np.NaN,
                return_train_score=False,
            )

            x_train, y_train = DENSITY_CLASSES[density_name](**density_params).simulate(
                n_samples=n_datapoints
            )
            # Arma Jump has a dimension different from the others
            if density_name == "ArmaJump":
                x_train = np.expand_dims(x_train, axis=1)
                y_train = np.expand_dims(y_train, axis=1)

            cv.fit(x_train, y_train)
            df = pandas.DataFrame(cv.cv_results_)
            print(df)
            df.to_csv(
                os.path.join(
                    results_dir,
                    "results_{}_{}.csv".format(
                        estimator["estimator_name"], density_name
                    ),
                )
            )

        df = pandas.DataFrame()
        for density, _ in density_list:
            temp = pandas.read_csv(
                os.path.join(
                    results_dir,
                    "results_{}_{}.csv".format(estimator["estimator_name"], density),
                )
            )
            temp = temp.join(pandas.Series([density] * len(temp), name="density"))
            temp = temp.join(
                pandas.Series(estimator["estimator_name"] * len(temp), name="estimator")
            )
            df = df.append(temp)

        df.to_csv(
            os.path.join(
                results_dir, "results_{}.csv".format(estimator["estimator_name"])
            )
        )
