import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions
from sklearn.model_selection import GridSearchCV
import pandas
from config import DATA_DIR
import numpy as np
from estimators import BayesianNFEstimator
from estimators import MaximumLikelihoodNFEstimator
from cde.density_simulation import SkewNormal, GaussianMixture, ArmaJump, EconDensity
from evaluation.scorers import bayesian_log_likelihood_score, mle_log_likelihood_score

TRAINING_EPOCHS = 2000
N_DATAPOINTS = 1000
densities = [
    (SkewNormal, {"random_seed": 22}),
    (EconDensity, {"random_seed": 22, "std": 1, "heteroscedastic": True}),
    (
        GaussianMixture,
        {
            "random_seed": 22,
            "n_kernels": 10,
            "ndim_x": 1,
            "ndim_y": 1,
            "means_std": 1.5,
        },
    ),
    (
        ArmaJump,
        {"random_seed": 22, "c": 0.1, "arma_a1": 0.9, "std": 0.05, "jump_prob": 0.05},
    ),
]

param_grid_mle = {
    "flow_types": [("radial", "radial", "radial")],
    "trainable_base_dist": [True],
    "hidden_sizes": [(10,), (7, 7), (20,)],
    "activation": ["tanh"],
    "x_noise_std": [0.0, 0.1, 0.2, 0.4],
    "y_noise_std": [0.0, 0.01, 0.1, 0.2],
    "learning_rate": [2e-2, 3e-3],
}

param_grid_bayesian = {
    "flow_types": [("radial", "radial", "radial")],
    "trainable_base_dist": [True],
    "hidden_sizes": [(10,), (7, 7), (20,)],
    "activation": ["tanh"],
    "x_noise_std": [0.0, 0.1, 0.2, 0.4],
    "y_noise_std": [0.0, 0.01, 0.1, 0.2],
    "learning_rate": [2e-2, 3e-3],
    "kl_weight_scale": [0.1, 0.001],
}

est_list = [
    {
        "estimator": BayesianNFEstimator,
        "estimator_name": "bayesian",
        "scoring_fn": bayesian_log_likelihood_score,
        "build_fn": BayesianNFEstimator.build_function,
        "param_grid": param_grid_bayesian,
    },
    {
        "estimator": MaximumLikelihoodNFEstimator,
        "estimator_name": "mle",
        "scoring_fn": mle_log_likelihood_score,
        "build_fn": MaximumLikelihoodNFEstimator.build_function,
        "param_grid": param_grid_mle,
    },
]
for est in est_list:
    for density, density_params in densities:
        model = tf.keras.wrappers.scikit_learn.KerasRegressor(
            build_fn=est["build_fn"], epochs=TRAINING_EPOCHS, verbose=0
        )
        cv = GridSearchCV(
            model,
            param_grid=est["param_grid"],
            cv=10,
            scoring=est["scoring_fn"],
            n_jobs=-1,
            verbose=2,
            error_score=np.NaN,
        )

        x_train, y_train = density(**density_params).simulate(n_samples=N_DATAPOINTS)
        # Arma Jump has a dimension different from the others
        if density == ArmaJump:
            x_train = np.expand_dims(x_train, axis=1)
            y_train = np.expand_dims(y_train, axis=1)

        cv.fit(x_train, y_train)
        pandas.DataFrame(cv.cv_results_).to_csv(
            os.path.join(
                DATA_DIR,
                "results_{}_{}.csv".format(est["estimator_name"], density.__name__),
            )
        )

    df = pandas.DataFrame()
    for density, _ in densities:
        temp = pandas.read_csv(
            os.path.join(
                DATA_DIR,
                "results_{}_{}.csv".format(est["estimator_name"], density.__name__),
            )
        )
        temp = temp.join(pandas.Series([density.__name__] * len(temp), name="density"))
        df = df.append(temp)

    df.to_csv(os.path.join(DATA_DIR, "results_{}.csv".format(est["estimator_name"])))
