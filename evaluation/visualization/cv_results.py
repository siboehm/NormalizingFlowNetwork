from config import DATA_DIR
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cluster_data_dir = os.path.join(DATA_DIR, "cluster/")
plots_dir = os.path.join(DATA_DIR, "visualization/")
score_columns = ["rank_test_score", "mean_test_score", "std_test_score"]
mle_columns = ["param_hidden_sizes", "param_x_noise_std", "param_y_noise_std"]
bayes_columns = [
    "param_hidden_sizes",
    "param_x_noise_std",
    "param_y_noise_std",
    "param_learning_rate",
    "param_kl_weight_scale",
    "param_prior_scale",
]


def output_overview(df, file, result_dir):
    filename = file.name.split(".")[0]
    columns = mle_columns if "mle" in filename else bayes_columns

    top_10 = df.sort_values("rank_test_score")[score_columns + columns].head(10)
    # top_10.to_latex(os.path.join(result_dir, "{}_top_10.tex".format(filename)))

    html = "<h1>{}</h1><h3>({} configurations total)</h3>".format(filename, len(df))
    html += top_10.to_html()

    for column in columns:
        html += "<h2>{}</h2>".format(column)
        html += df.groupby(column)[[column, "rank_test_score", "mean_test_score"]].mean().to_html()

    with open(os.path.join(result_dir, "{}_overview.html".format(filename)), "w") as f:
        f.write(html)

    fig, axarr = plt.subplots(1, len(columns), figsize=(15, 7))
    for i, column in enumerate(columns):
        df.boxplot(ax=axarr[i], column=["mean_test_score"], by=column, grid=False)

    fig.suptitle("Marginalized test scores for every hyperparameter")
    plt.savefig(os.path.join(result_dir, filename + "_" + "hyperparams" + ".png"))


def plot_heatplots(df, file, result_dir):
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

            # hack, there are always a few rogue values and those mess up the plots
            if "SkewNormal" in file.name:
                prev = len(df)
                df = df[df.mean_test_score > -2]
                assert prev < len(df) + 15
            output_overview(df, file, result_dir)
            plot_heatplots(df, file, result_dir)
