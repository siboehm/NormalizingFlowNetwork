from multiprocessing import Manager
from evaluation.empirical_eval.async_executor import AsyncExecutor, LoopExecutor
from estimators import ESTIMATORS
import numpy as np
import tensorflow as tf
import pandas as pd
import copy
from pprint import pprint
from ml_logger import logger


def run_benchmark_train_test_fit_cv(
    dataset,
    model_dict,
    seed=27,
    n_jobs_inner=-1,
    n_jobc_outer=1,
    n_train_valid_splits=1,
    shuffle_splits=True,
    n_eval_seeds=1,
    n_folds=5,
):
    rds = np.random.RandomState(seed)
    eval_seeds = list(rds.randint(0, 10 ** 7, size=n_eval_seeds))

    logger.log(
        "\n------------------  empirical benchmark with %s ----------------------" % str(dataset)
    )

    for model_key in model_dict:
        model_dict[model_key].update({"n_dims": dataset.ndim_y})

    # run experiments
    cv_result_dicts = []

    datasets = zip(
        *dataset.get_train_valid_splits(
            valid_portion=0.2,
            n_splits=n_train_valid_splits,
            shuffle=shuffle_splits,
            random_state=rds,
        )
    )

    for i, (X_train, Y_train, X_valid, Y_valid) in enumerate(datasets):
        logger.log("--------  train-valid split %i --------" % i)

        manager = Manager()
        cv_result_dict = manager.dict()

        def _fit_by_cv_and_eval(estimator_key, conf_dict):
            logger.log("evaluating %s parameters with %i seeds" % (estimator_key, len(eval_seeds)))
            scores = _evaluate_params(
                estimator_key, conf_dict, X_train, Y_train, X_valid, Y_valid, seeds=eval_seeds
            )

            cv_result_dict[estimator_key] = {
                "selected_params": conf_dict,
                "scores": scores,
                "eval_seeds": eval_seeds,
            }
            logger.log("evaluation scores for %s: %s" % (estimator_key, str(scores)))

        executor = AsyncExecutor(n_jobs=n_jobc_outer)
        executor.run(_fit_by_cv_and_eval, model_dict.keys(), model_dict.values())

        cv_result_dicts.append(dict(cv_result_dict))

    pprint(cv_result_dicts)

    # rearrange results as pandas df
    final_results_dict = {"scores_mean": [], "scores_std": [], "dataset": []}
    for estimator_key in model_dict.keys():
        scores = []
        for result_dict in cv_result_dicts:
            scores.extend(result_dict[estimator_key]["scores"])

        final_results_dict["scores_mean"].append(np.mean(scores))
        final_results_dict["scores_std"].append(np.std(scores))
        final_results_dict["dataset"].append(str(dataset))

    df = pd.DataFrame.from_dict(data=final_results_dict, orient="columns")
    df.index = list(model_dict.keys())

    logger.log("\n" + str(df))
    return df


""" helpers """


def _evaluate_params(estimator_key, param_dict, X_train, Y_train, X_valid, Y_valid, seeds):
    estimator_class = ESTIMATORS[estimator_key.replace("_MAP", "")]
    eval_scores = []

    def _eval_with_seed(seed):
        config = tf.ConfigProto(
            device_count={"CPU": 1}, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
        )
        with tf.Session(config=config):
            param_dict_local = copy.copy(param_dict)
            param_dict_local["random_seed"] = seed
            param_dict_local["kl_weight_scale"] = 1.0 / X_train.shape[0]
            param_dict_local["map_mode"] = "MAP" in estimator_key
            # param_dict_local["name"] += str(seed)
            est = estimator_class(**param_dict_local)
            est.fit(X_train, Y_train, epochs=2000, verbose=0)
            score = est.score(X_valid, Y_valid)
            eval_scores.append(score)

        tf.reset_default_graph()

    executor = LoopExecutor()
    executor.run(_eval_with_seed, seeds)

    return eval_scores
