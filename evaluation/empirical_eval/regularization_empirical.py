from evaluation.empirical_eval.experiment_util import run_benchmark_train_test_fit_cv
import evaluation.empirical_eval.datasets as datasets
from ml_logger import logger
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python import tf2

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/local"))
EXP_PREFIX = "regularization_empirical"

model_dict = {
    "bayesian_NFN_MAP": {
        "n_flows": 10,
        "hidden_sizes": (32, 32),
        "learning_rate": 2e-2,
        "map_mode": True,
    },
    "bayesian_NFN": {
        "n_flows": 10,
        "hidden_sizes": (32, 32),
        "learning_rate": 2e-2,
        "map_mode": False,
    },
    "bayesian_MDN_MAP": {
        "n_centers": 10,
        "hidden_sizes": (32, 32),
        "learning_rate": 2e-2,
        "map_mode": True,
    },
    "bayesian_MDN": {
        "n_centers": 10,
        "hidden_sizes": (32, 32),
        "learning_rate": 2e-2,
        "map_mode": False,
    },
    "bayesian_KMN_MAP": {
        "n_centers": 50,
        "hidden_sizes": (32, 32),
        "learning_rate": 2e-2,
        "map_mode": True,
    },
    "bayesian_KMN": {
        "n_centers": 50,
        "hidden_sizes": (32, 32),
        "learning_rate": 2e-2,
        "map_mode": False,
    },
}


def experiment():
    logger.configure(log_directory=os.path.join(DATA_DIR), prefix=EXP_PREFIX, color="green")

    # 1) EUROSTOXX
    dataset = datasets.EuroStoxx50()

    result_df = run_benchmark_train_test_fit_cv(
        dataset,
        model_dict,
        n_train_valid_splits=3,
        n_eval_seeds=5,
        shuffle_splits=False,
        n_folds=5,
        seed=22,
        n_jobs_inner=1,
        n_jobc_outer=3,
    )

    # 2) NYC Taxi
    for n_samples in [10000]:
        dataset = datasets.NCYTaxiDropoffPredict(n_samples=n_samples)

    df = run_benchmark_train_test_fit_cv(
        dataset,
        model_dict,
        n_train_valid_splits=3,
        n_eval_seeds=5,
        shuffle_splits=True,
        n_folds=5,
        seed=22,
        n_jobs_inner=-1,
        n_jobc_outer=3,
    )
    result_df = pd.concat([result_df, df], ignore_index=True)

    # 3) UCI
    for dataset_class in [datasets.BostonHousing, datasets.Conrete, datasets.Energy]:
        dataset = dataset_class()
        df = run_benchmark_train_test_fit_cv(
            dataset,
            model_dict,
            n_train_valid_splits=3,
            n_eval_seeds=5,
            shuffle_splits=True,
            n_folds=5,
            seed=22,
            n_jobs_inner=-1,
            n_jobc_outer=3,
        )
        result_df = pd.concat([result_df, df], ignore_index=True)

    logger.log("\n", str(result_df))
    logger.log("\n", result_df.tolatex())


if __name__ == "__main__":
    experiment()
