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


parser = argparse.ArgumentParser()
parser.add_argument(
    "run_config", help="name of the run configuation stored in data/local/"
)
args = parser.parse_args()

config_file_path = os.path.join(DATA_DIR, "local/", args.run_config)
with open(config_file_path, "r") as f:
    run_config = json.load(f)
config_dir_name = (
    args.run_config.split(".")[0] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M")
)
config_dir_path = os.path.join(DATA_DIR, "local/", config_dir_name)
os.mkdir(config_dir_path)

DENSITIES = run_config["density_configs"]
N_TRAINING_EPOCHS = run_config["n_training_epochs"]
N_FOLDS = run_config["n_folds"]
N_DATAPOINTS = run_config["n_datapoints"]
ESTIMATOR_LIST = [
    {
        "estimator": BayesianNFEstimator,
        "estimator_name": "bayesian",
        "scoring_fn": bayesian_log_likelihood_score,
        "build_fn": BayesianNFEstimator.build_function,
        "param_grid": run_config["param_grid_bayesian"],
    },
    {
        "estimator": MaximumLikelihoodNFEstimator,
        "estimator_name": "mle",
        "scoring_fn": mle_log_likelihood_score,
        "build_fn": MaximumLikelihoodNFEstimator.build_function,
        "param_grid": run_config["param_grid_mle"],
    },
]

run_configuation(
    estimator_list=ESTIMATOR_LIST,
    density_list=DENSITIES,
    n_epochs=N_TRAINING_EPOCHS,
    n_folds=N_FOLDS,
    n_datapoints=N_DATAPOINTS,
    results_dir=config_dir_path,
)
