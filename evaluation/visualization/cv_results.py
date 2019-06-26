from config import DATA_DIR
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

cluster_data_dir = os.path.join(DATA_DIR, "cluster/")
plots_dir = os.path.join(DATA_DIR, "visualization/")
score_columns = ["rank_test_score", "mean_test_score", "std_test_score"]
mle_columns = ["param_hidden_sizes", "param_learning_rate", "param_noise_reg"]
bayes_columns = [
    "param_hidden_sizes",
    "param_learning_rate",
    "param_prior_scale",
    "param_map_mode",
    "param_noise_reg",
]


def output_overview(df, file, result_dir):
    filename = file.name.split(".")[0]
    columns = mle_columns if "mle" in filename else bayes_columns
    columns = [x for x in columns if x in df.columns]

    top_10 = df.sort_values("rank_test_score")[score_columns + columns].head(10)
    # top_10.to_latex(os.path.join(result_dir, "{}_top_10.tex".format(filename)))

    html = "<h1>{}</h1><h3>({} configurations total)</h3>".format(filename, len(df))
    html += top_10.to_html()

    for column in columns:
        html += "<h2>{}</h2>".format(column)
        html += df.groupby(column)[[column, "rank_test_score", "mean_test_score"]].mean().to_html()

    with open(os.path.join(result_dir, "{}_overview.html".format(filename)), "w") as f:
        f.write(html)

    fig, axarr = plt.subplots(1, len(columns), figsize=(16, 7))
    for i, column in enumerate(columns):
        df.boxplot(ax=axarr[i], column=["mean_test_score"], by=column, grid=False)

    fig.suptitle("Marginalized test scores for every hyperparameter")
    plt.savefig(os.path.join(result_dir, filename + "_" + "hyperparams" + ".png"))


def output_metric_scores(df, file, result_dir):
    densities = df["density"].unique()
    param_columns = mle_columns if "mle" in file.name else bayes_columns
    tuned_params = [
        param for param in param_columns if param in df.columns and len(df[param].unique()) > 1
    ]
    test_score_columns = [
        column
        for column in df.columns
        if column.startswith("split") and column.endswith("_test_score")
    ]

    for param in tuned_params:
        plot_single_param(densities, df, file, param, result_dir, test_score_columns)

    plot_single_param(
        densities,
        df.loc[df["param_map_mode"] == True],
        file,
        "param_noise_reg",
        result_dir,
        test_score_columns,
        name_prefix="map_reg",
    )
    plot_single_param(
        densities,
        df.loc[df["param_map_mode"] == False],
        file,
        "param_noise_reg",
        result_dir,
        test_score_columns,
        name_prefix="bayes_reg",
    )


def plot_single_param(
    densities, df, file, param, result_dir, test_score_columns, name_prefix="", log_scale_y=False
):
    if len(densities) == 1:
        layout = (1, 1)
    else:
        layout = (len(densities) // 2 + 1, 2)
    fig, axarr = plt.subplots(*layout, figsize=(11 * layout[1], 5 * layout[0]))
    axarr = [axarr] if len(densities) == 1 else axarr.flatten()

    n_curves_to_plot = len(df[param].unique())
    for i, density in enumerate(densities):
        color_iter = iter(cm.gist_rainbow(np.linspace(0, 1, n_curves_to_plot)))
        for param_instance in df[param].unique():
            sub_df = df.loc[(df["density"] == density) & (df[param] == param_instance)]

            n_datapoints = sorted(sub_df["n_datapoints"].unique())
            means = np.array([], dtype=np.float32)
            stds = np.array([], dtype=np.float32)
            for n_data in n_datapoints:
                scores = []
                for c in test_score_columns:
                    scores += list(sub_df.loc[sub_df["n_datapoints"] == n_data][c].values)
                scores = np.array(scores, dtype=np.float32)
                means = np.append(means, scores.mean())
                stds = np.append(stds, scores.std())

            c = next(color_iter)

            axarr[i].plot(
                n_datapoints,
                means,
                color=c,
                label="{}: {}".format(param.replace("param_", ""), param_instance),
            )
            axarr[i].fill_between(n_datapoints, means - stds, means + stds, alpha=0.2, color=c)

        axarr[i].set_xlabel("n_observations")
        axarr[i].set_ylabel("score")
        axarr[i].set_title(density)
        axarr[i].legend()
        axarr[i].set_xscale("log")
        if log_scale_y:
            axarr[i].set_yscale("log")
    plt.savefig(
        os.path.join(
            result_dir,
            file.name.split(".")[0] + "_" + name_prefix + "_metric_scores_" + param + ".png",
        )
    )


def plot_reg(df, file, result_dir):
    densities = df["density"].unique()
    estimators = df["estimator"].unique()
    df["param_kl_weight_scale"] = df.param_kl_weight_scale * df.n_datapoints

    kl_params = df["param_kl_weight_scale"].unique()
    map_params = df["param_map_mode"].unique()

    test_score_columns = [
        column
        for column in df.columns
        if column.startswith("split") and column.endswith("_test_score")
    ]
    layout = (1, 2)
    n_curves_to_plot = len(kl_params) * len(map_params)

    for estimator in estimators:
        fig, axarr = plt.subplots(*layout, figsize=(11 * layout[1], 5 * layout[0]))
        axarr = [axarr] if len(densities) == 1 else axarr.flatten()
        for i, density in enumerate(densities):
            color_iter = iter(cm.gist_rainbow(np.linspace(0, 1, n_curves_to_plot)))
            for kl in kl_params:
                for map in map_params:
                    sub_df = df.loc[
                        (df["density"] == density)
                        & (df["estimator"] == estimator)
                        & (df["param_map_mode"] == map)
                        & (df["param_kl_weight_scale"] == kl)
                    ]
                    n_datapoints = sorted(sub_df["n_datapoints"].unique())
                    means = np.array([], dtype=np.float32)
                    stds = np.array([], dtype=np.float32)

                    for n_data in n_datapoints:
                        scores = []
                        for c in test_score_columns:
                            scores += list(sub_df.loc[sub_df["n_datapoints"] == n_data][c].values)
                        scores = np.array(scores, dtype=np.float32)
                        means = np.append(means, scores.mean())
                        stds = np.append(stds, scores.std())

                    c = next(color_iter)

                    label = "{}: kl_scale: {}, map_mode: {}".format(estimator, kl, map)
                    axarr[i].plot(n_datapoints, means, color=c, label=label)
                    axarr[i].fill_between(
                        n_datapoints, means - stds, means + stds, alpha=0.2, color=c
                    )

                axarr[i].set_xlabel("n_observations")
                axarr[i].set_ylabel("score")
                axarr[i].set_title(density)
                axarr[i].legend()
                axarr[i].set_xscale("log")

        plt.savefig(
            os.path.join(
                result_dir, file.name.split(".")[0] + "_" + "noise_reg_{}.png".format(estimator)
            )
        )


def plot_map_mle(df, file, result_dir):
    densities = df["density"].unique()
    test_score_columns = [
        column
        for column in df.columns
        if column.startswith("split") and column.endswith("_test_score")
    ]
    param_columns = mle_columns if "mle" in file.name else bayes_columns
    if len(densities) == 1:
        layout = (1, 1)
    else:
        layout = (len(densities) // 2 + 1, 2)

    fig, axarr = plt.subplots(*layout, figsize=(11 * layout[1], 5 * layout[0]))
    axarr = [axarr] if len(densities) == 1 else axarr.flatten()
    n_curves_to_plot = 2

    for i, density in enumerate(densities):
        color_iter = iter(cm.gist_rainbow(np.linspace(0, 1, n_curves_to_plot)))

        for name, estimator in [("MAP", "bayesian"), ("MLE", "mle")]:
            if estimator == "bayesian":
                sub_df = df.loc[
                    (df["density"] == density)
                    & (df["estimator"] == estimator)
                    & (df["param_map_mode"])
                    & (df["param_noise_reg"] == "['fixed_rate', 0.0]")
                ]
            else:
                sub_df = df.loc[
                    (df["density"] == density)
                    & (df["estimator"] == estimator)
                    & (df["param_noise_reg"] == "['fixed_rate', 0.0]")
                ]
            n_datapoints = sorted(sub_df["n_datapoints"].unique())
            means = np.array([], dtype=np.float32)
            stds = np.array([], dtype=np.float32)

            for n_data in n_datapoints:
                scores = []
                for c in test_score_columns:
                    scores += list(sub_df.loc[sub_df["n_datapoints"] == n_data][c].values)
                scores = np.array(scores, dtype=np.float32)
                means = np.append(means, scores.mean())
                stds = np.append(stds, scores.std())

            c = next(color_iter)

            axarr[i].plot(n_datapoints, means, color=c, label=name)
            axarr[i].fill_between(n_datapoints, means - stds, means + stds, alpha=0.2, color=c)

            axarr[i].set_xlabel("n_observations")
            axarr[i].set_ylabel("score")
            axarr[i].set_title(density)
            axarr[i].legend()
            axarr[i].set_xscale("log")

        plt.savefig(os.path.join(result_dir, file.name.split(".")[0] + "_" + "map_vs_mle.png"))


def plot_best(df, file, result_dir):
    densities = df["density"].unique()
    test_score_columns = [
        column
        for column in df.columns
        if column.startswith("split") and column.endswith("_test_score")
    ]
    param_columns = mle_columns if "mle" in file.name else bayes_columns
    if len(densities) == 1:
        layout = (1, 1)
    else:
        layout = (len(densities) // 2 + 1, 2)

    fig, axarr = plt.subplots(*layout, figsize=(11 * layout[1], 5 * layout[0]))
    axarr = [axarr] if len(densities) == 1 else axarr.flatten()
    n_curves_to_plot = 3

    for i, density in enumerate(densities):
        color_iter = iter(cm.gist_rainbow(np.linspace(0, 1, n_curves_to_plot)))
        best_bay = df.loc[
            (df["density"] == density)
            & (df["estimator"] == "bayesian")
            & (df["param_map_mode"] == False)
            & (df["n_datapoints"] == 800)
        ].sort_values(by="mean_test_score", ascending=False)
        best_map = df.loc[
            (df["density"] == density)
            & (df["estimator"] == "bayesian")
            & (df["param_map_mode"] == True)
            & (df["n_datapoints"] == 800)
        ].sort_values(by="mean_test_score", ascending=False)
        best_mle = df.loc[
            (df["density"] == density) & (df["estimator"] == "mle") & (df["n_datapoints"] == 800)
        ].sort_values(by="mean_test_score", ascending=False)

        for name, map_mode, estimator, best in [
            ("full bayesian", False, "bayesian", best_bay),
            ("MAP", True, "bayesian", best_map),
            ("MLE", False, "mle", best_mle),
        ]:
            if estimator == "bayesian":
                sub_df = df.loc[
                    (df["density"] == density)
                    & (df["estimator"] == estimator)
                    & (df["param_map_mode"] == map_mode)
                    & (df["param_noise_reg"] == best["param_noise_reg"].values[0])
                ]
            else:
                sub_df = df.loc[
                    (df["density"] == density)
                    & (df["estimator"] == estimator)
                    & (df["param_noise_reg"] == best["param_noise_reg"].values[0])
                ]
            n_datapoints = sorted(sub_df["n_datapoints"].unique())
            means = np.array([], dtype=np.float32)
            stds = np.array([], dtype=np.float32)

            for n_data in n_datapoints:
                scores = []
                for c in test_score_columns:
                    scores += list(sub_df.loc[sub_df["n_datapoints"] == n_data][c].values)
                scores = np.array(scores, dtype=np.float32)
                means = np.append(means, scores.mean())
                stds = np.append(stds, scores.std())

            c = next(color_iter)

            actual_columns = [p for p in param_columns if p in df.columns]
            label = " ".join(
                [name, ":"]
                + [
                    "{}: {},".format(col.replace("param_", ""), best[col].values[0])
                    for col in actual_columns
                ]
            )
            axarr[i].plot(n_datapoints, means, color=c, label=label)
            axarr[i].fill_between(n_datapoints, means - stds, means + stds, alpha=0.2, color=c)

            axarr[i].set_xlabel("n_observations")
            axarr[i].set_ylabel("score")
            axarr[i].set_title(density)
            axarr[i].legend()
            axarr[i].set_xscale("log")

        plt.savefig(os.path.join(result_dir, file.name.split(".")[0] + "_" + "best_scorers.png"))


def plot_noise_heatplots(df, file, result_dir):
    filename = file.name.split(".")[0]
    x_noise_vals = list(reversed(sorted(df.param_x_noise_std.unique())))
    y_noise_vals = sorted(df.param_y_noise_std.unique())
    result_grid = np.empty((len(x_noise_vals), len(y_noise_vals)))
    fig, axarr = plt.subplots(1, 1, figsize=(6, 4))
    for i, x_noise_std in enumerate(x_noise_vals):
        for j, y_noise_std in enumerate(y_noise_vals):
            sub_df = df.loc[
                (df["param_x_noise_std"] == x_noise_std) & (df["param_y_noise_std"] == y_noise_std)
            ]
            result_grid[i, j] = sub_df["mean_test_score"].mean()
    im = axarr.imshow(result_grid)
    # annotate pixels
    for i, x_noise_std in enumerate(x_noise_vals):
        for j, y_noise_std in enumerate(y_noise_vals):
            axarr.text(j, i, "%.3f" % result_grid[i, j], ha="center", va="center", color="w")
    axarr.set_ylabel("x-noise std")
    axarr.set_xlabel("y-noise std")
    axarr.set_yticks(np.arange(len(x_noise_vals)))
    axarr.set_xticks(np.arange(len(y_noise_vals)))
    axarr.set_yticklabels([str(val) for val in x_noise_vals])
    axarr.set_xticklabels([str(val) for val in y_noise_vals])
    axarr.set_title(filename)
    cbar = axarr.figure.colorbar(im, ax=axarr, shrink=0.8)
    cbar.ax.set_ylabel("Log Likelihood", rotation=-90, va="bottom")
    plt.savefig(os.path.join(result_dir, filename + "_xy_noise.png"))
    plt.close()


def plot_kl_weight_scale_heat(df, file, result_dir):
    filename = file.name.split(".")[0]
    kl_weight_scales = list(reversed(sorted(df.param_kl_weight_scale.unique())))
    prior_scales = sorted(df.param_prior_scale.unique())
    result_grid = np.empty((len(kl_weight_scales), len(prior_scales)))
    fig, axarr = plt.subplots(1, 1, figsize=(6, 4))
    for i, kl_weight_scale in enumerate(kl_weight_scales):
        for j, prior_scale in enumerate(prior_scales):
            sub_df = df.loc[
                (df["param_kl_weight_scale"] == kl_weight_scale)
                & (df["param_prior_scale"] == prior_scale)
            ]
            result_grid[i, j] = sub_df["mean_test_score"].mean()
    im = axarr.imshow(result_grid)
    # annotate pixels
    for i, kl_weight_scale in enumerate(kl_weight_scales):
        for j, prior_scale in enumerate(prior_scales):
            axarr.text(j, i, "%.3f" % result_grid[i, j], ha="center", va="center", color="w")
    axarr.set_ylabel("kl weight scale")
    axarr.set_xlabel("prior scale")
    axarr.set_yticks(np.arange(len(kl_weight_scales)))
    axarr.set_xticks(np.arange(len(prior_scales)))
    axarr.set_yticklabels([str(val) for val in kl_weight_scales])
    axarr.set_xticklabels([str(val) for val in prior_scales])
    axarr.set_title(filename)
    cbar = axarr.figure.colorbar(im, ax=axarr, shrink=0.8)
    cbar.ax.set_ylabel("Log Likelihood", rotation=-90, va="bottom")
    plt.savefig(os.path.join(result_dir, filename + "_kl_scale_prior.png"))
    plt.close()


def plot_cv_results():
    data_folders = [
        f for f in os.scandir(cluster_data_dir) if f.is_dir() and not f.path.endswith("logs")
    ]

    for folder in data_folders:
        result_dir = os.path.join(plots_dir, folder.name)
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        data_files = [
            file
            for file in os.scandir(folder.path)
            if file.is_file() and file.name.endswith(".csv")
        ]

        for file in data_files:
            df = pd.read_csv(file.path)

            if ("06-17_22-33" in result_dir) and file.name == "results.csv" and False:
                output_metric_scores(df, file, result_dir)
                plot_best(df, file, result_dir)
                plot_map_mle(df, file, result_dir)

            if ("bayes_reg_all" in result_dir) and file.name == "results.csv":
                plot_reg(df, file, result_dir)
