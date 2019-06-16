import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from datetime import datetime
from config import DATA_DIR
from estimators import BayesianNFEstimator
from estimators import MaximumLikelihoodNFEstimator
from evaluation.scorers import bayesian_log_likelihood_score, mle_log_likelihood_score
from evaluation.config_runner import run_configuation
from collections.abc import Iterable


def load_config_file(config_filename):
    config_file_path = os.path.join(DATA_DIR, "local/", config_filename)
    with open(config_file_path, "r") as f:
        return json.load(f)


def create_results_dir(config_filename):
    config_dir_name = (
        config_filename.split(".")[0] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M")
    )
    config_path = os.path.join(DATA_DIR, "local/", config_dir_name)
    os.mkdir(config_path)
    return config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_config", help="name of the run configuation stored in data/local/")
    parser.add_argument('--n_jobs', type=int, default=-1)
    args = parser.parse_args()

    run_config = load_config_file(args.run_config)
    config_dir_path = create_results_dir(args.run_config)

    # write the file into the newly created folder for reference
    with open(os.path.join(config_dir_path, "run_config.json"), "w") as f:
        json.dump(run_config, f)

    # collect the configurations for the estimators
    ESTIMATOR_LIST = []
    if run_config.get("param_grid_bayesian"):
        ESTIMATOR_LIST.append(
            {
                "estimator": BayesianNFEstimator,
                "estimator_name": "bayesian",
                "scoring_fn": bayesian_log_likelihood_score,
                "build_fn": BayesianNFEstimator.build_function,
                "param_grid": run_config["param_grid_bayesian"],
            }
        )
    if run_config.get("param_grid_mle"):
        ESTIMATOR_LIST.append(
            {
                "estimator": MaximumLikelihoodNFEstimator,
                "estimator_name": "mle",
                "scoring_fn": mle_log_likelihood_score,
                "build_fn": MaximumLikelihoodNFEstimator.build_function,
                "param_grid": run_config["param_grid_mle"],
            }
        )

    n_datapoints = run_config["n_datapoints"]
    n_datapoints_list = n_datapoints if isinstance(n_datapoints, Iterable) else [n_datapoints]

    run_configuation(
        estimator_list=ESTIMATOR_LIST,
        density_list=run_config["density_configs"],
        n_epochs=run_config["n_training_epochs"],
        n_folds=run_config["n_folds"],
        n_datapoints_list=n_datapoints_list,
        results_dir=config_dir_path,
        n_jobs=args.n_jobs,
    )
