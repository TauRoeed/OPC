import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import gaussian_kde, lognorm
import pandas as pd

import plotly.express as px
import pandas as pd
import plotly.io as pio

# optional: set a global default once in your module
pio.renderers.default = "browser"  # or "vscode", "svg", etc.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def unpack_scores_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw study df (with 'user_attrs_scores_dict') and return
    a new DataFrame with flattened score_* columns.
    """
    if "user_attrs_scores_dict" not in df.columns:
        return df.copy()

    def parse_scores_dict(s: str):
        import numpy as np
        return eval(s, {"np": np})

    df = df.copy()
    df["scores_dict"] = df["user_attrs_scores_dict"].apply(parse_scores_dict)
    scores_df = pd.json_normalize(df["scores_dict"])
    scores_df.columns = [f"score_{c}" for c in scores_df.columns]

    full_df = pd.concat([df.drop(columns=["user_attrs_scores_dict"]), scores_df], axis=1)
    return full_df


def plot_ope_study_diagnostics(
    df: pd.DataFrame,
    score_cols: list[str] | None = None,
    true_col: str = "user_attrs_actual_reward",   # <<<<<<<< UPDATED HERE
    est_col: str = "user_attrs_r_hat",            # your DR-shrink estimate
):
    """
    Given the study df, unpack scores and produce diagnostic plots comparing
    score_* metrics to actual rewards and estimation errors.

    true_col: what you're trying to approximate (ground truth).
    est_col:  your estimator (DR shrink by default).
    """

    full_df = unpack_scores_dict(df)

    # --- derived metrics ---
    full_df["true_value"] = full_df[true_col]
    full_df["est_value"] = full_df[est_col]
    full_df["error"] = full_df["est_value"] - full_df["true_value"]
    full_df["abs_error"] = full_df["error"].abs()
    full_df["regret"] = full_df["true_value"] - full_df["est_value"]

    # choose default score columns if not provided
    if score_cols is None:
        candidates = [
            "score_dr_naive_mean",
            "score_dr_boot_mean",
            "score_ipw_boot_mean",
            "score_score_naive_minus_cv_uniform",
            "score_score_naive_minus_cv_exp",
        ]
        score_cols = [c for c in candidates if c in full_df.columns]
        if not score_cols:
            score_cols = [c for c in full_df.columns if c.startswith("score_")]

    # -------------------------------
    # 1. Estimator vs ground truth
    # -------------------------------
    plt.figure()
    plt.scatter(full_df["true_value"], full_df["est_value"], alpha=0.7)
    lo = min(full_df["true_value"].min(), full_df["est_value"].min())
    hi = max(full_df["true_value"].max(), full_df["est_value"].max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel(f"True value ({true_col})")
    plt.ylabel(f"Estimated value ({est_col})")
    plt.title("Estimated vs True Value")
    plt.tight_layout()

    # -------------------------------
    # 2. For each score: vs truth & vs abs error
    # -------------------------------
    for col in score_cols:

        # score vs actual reward
        plt.figure()
        plt.scatter(full_df[col], full_df["true_value"], alpha=0.7)
        plt.xlabel(col)
        plt.ylabel(true_col)
        plt.title(f"{col} vs {true_col}")
        plt.tight_layout()
        plot_ranked_reward_curve(full_df[col].values, full_df["true_value"].values, full_df[col].values, window_size=1)
    # -------------------------------
    # 3. Correlations
    # -------------------------------
    corr_true = []
    corr_neg_abs_err = []

    for col in score_cols:
        corr_true.append(full_df[[col, "true_value"]].corr().iloc[0, 1])
        corr_neg_abs_err.append(
            full_df[[col, "abs_error"]].corr().iloc[0, 1] * -1.0
        )

    x = np.arange(len(score_cols))

    plt.figure()
    plt.bar(x - 0.15, corr_true, width=0.3, label=f"corr(score, {true_col})")
    plt.bar(x + 0.15, corr_neg_abs_err, width=0.3, label="corr(score, -abs_error)")
    plt.xticks(x, score_cols, rotation=45, ha="right")
    plt.ylabel("Correlation")
    plt.title(f"Correlation of Scoring Rules with {true_col} / -Abs Error")
    plt.legend()
    plt.tight_layout()

    return full_df


def compute_correlations(score, actual, est):
    return {
        "pearson_score_actual": np.corrcoef(score, actual)[0, 1],
        "spearman_score_actual": spearmanr(score, actual).correlation,
        "spearman_est_actual": spearmanr(est, actual).correlation,
    }


def dcg(relevances):
    rel = np.asarray(relevances)
    return np.sum((2**rel - 1) / np.log2(np.arange(1, len(rel)+1) + 1))


def ndcg(scores, rewards):
    order = np.argsort(scores)[::-1]
    ranked = rewards[order]
    ideal = dcg(np.sort(rewards)[::-1])
    return dcg(ranked) / ideal if ideal > 0 else 0.0


def compute_ndcg(score, actual, est):
    return {
        "ndcg_score_actual": ndcg(score, actual),
        "ndcg_est_actual": ndcg(est, actual),
    }


def compute_error_metrics(actual, est):
    err = actual - est
    return {
        "mean_error": err.mean(),
        "mae": np.abs(err).mean(),
        "rmse": np.sqrt((err**2).mean()),
        "std_error": err.std(),
        "max_error": err.max(),
        "min_error": err.min(),
    }


def plot_ess_heatmap_scatter(score, actual, ess):
    plt.figure(figsize=(8,6))
    sc = plt.scatter(
        score, actual,
        c=ess, cmap="viridis",
        s=40, alpha=0.7, edgecolor="none"
    )
    plt.colorbar(sc, label="ESS")
    plt.xlabel("Score")
    plt.ylabel("Actual Reward")
    plt.title("ESS Heatmap Scatter")
    plt.grid(True)
    plt.show()


def plot_ranked_reward_curve(score, actual, est, window_size = 9):
    order = np.argsort(score)[::-1]
    weights = np.ones(window_size) / window_size
    plt.figure(figsize=(8,5))
    plt.plot(np.convolve(actual[order], weights, mode='valid'), label="Actual Reward", linewidth=2)
    plt.plot(np.convolve(est[order], weights, mode='valid'), label="Estimated Reward", linewidth=2, alpha=0.7)
    plt.title("Ranked Reward Curve (Sorted by Score)")
    plt.xlabel("Rank")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_scatter(score, actual):
    plt.figure(figsize=(6,5))
    plt.scatter(score, actual, alpha=0.4)
    plt.xlabel("Score")
    plt.ylabel("Actual Reward")
    plt.title("Scatter: Score vs Actual Reward")
    plt.grid(True)
    plt.show()


def plot_hexbin(score, actual):
    plt.figure(figsize=(6,5))
    hb = plt.hexbin(score, actual, gridsize=40, cmap='viridis', mincnt=1)
    plt.colorbar(hb, label="Count")
    plt.xlabel("Score")
    plt.ylabel("Actual Reward")
    plt.title("Hexbin Plot")
    plt.show()


def plot_kde(values_list, labels):
    plt.figure(figsize=(6,5))
    for vals, label in zip(values_list, labels):
        kde = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 400)
        plt.plot(xs, kde(xs), linewidth=2, label=label)
    plt.legend(); plt.grid(True)
    plt.title("KDE Distributions")
    plt.show()


def plot_centered_kde(values_list, labels):
    plt.figure(figsize=(6,5))
    for vals, label in zip(values_list, labels):
        centered = vals - vals.mean()
        kde = gaussian_kde(centered)
        xs = np.linspace(centered.min(), centered.max(), 400)
        plt.plot(xs, kde(xs), linewidth=2, label=label)
    plt.legend(); plt.grid(True)
    plt.title("Centered KDE Distributions")
    plt.show()


def plot_log_kde_with_lognormal_fit(values_list, labels):
    eps = 1e-8
    plt.figure(figsize=(7,5))

    for vals, label in zip(values_list, labels):
        logvals = np.log(vals + eps)
        kde = gaussian_kde(logvals)
        xs = np.linspace(logvals.min(), logvals.max(), 400)
        plt.plot(xs, kde(xs), label=f"{label} Log-KDE", linewidth=2)

        # Fit lognormal PDF
        sigma, loc, scale = lognorm.fit(vals, floc=0)
        pdf = lognorm.pdf(np.exp(xs), sigma, loc=0, scale=scale) * np.exp(xs)
        plt.plot(xs, pdf, '--', linewidth=1.5, label=f"{label} LogNormal Fit")

    plt.legend(); plt.grid(True)
    plt.xlabel("log(Value)")
    plt.title("Log-KDE + Fitted Log-Normal PDF")
    plt.show()


def plot_calibration_curve(score, actual, n_bins=20):
    order = np.argsort(score)
    sorted_score = score[order]
    sorted_actual = actual[order]

    bins = np.array_split(np.arange(len(score)), n_bins)
    avg_pred = [sorted_score[b].mean() for b in bins]
    avg_actual = [sorted_actual[b].mean() for b in bins]

    plt.figure(figsize=(6,5))
    plt.plot(avg_pred, avg_actual, marker='o')
    lo, hi = min(avg_pred), max(avg_pred)
    plt.plot([lo, hi], [lo, hi], 'k--', label="Perfect calibration")
    plt.title("Calibration Curve")
    plt.xlabel("Mean Score (bin)")
    plt.ylabel("Mean Actual Reward (bin)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_error_hover(estimated_error, actual_error, ESS,
                     title="Interactive Error vs Actual Error (ESS-colored)",
                     x_label="Estimated Error",
                     y_label="Actual Error"):
    
    df_plot = pd.DataFrame({
        f"{x_label}": estimated_error,
        f"{y_label}": actual_error,
        "ESS": ESS
    })

    fig = px.scatter(
        df_plot,
        x=f"{x_label}",
        y=f"{y_label}",
        color="ESS",
        color_continuous_scale="Viridis",
        hover_data=[f"{x_label}", f"{y_label}", "ESS"],
        title=title,
    )

    fig.update_traces(marker=dict(size=9, opacity=0.8))
    fig.update_layout(
        width=900,
        height=700,
        coloraxis_colorbar=dict(title="ESS"),
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
    )

    # key change: specify renderer that doesn't need nbformat
    fig.show(renderer="browser")


def plot_error_plots(actual, est, score):
    err = actual - est

    # error histogram
    plt.figure(figsize=(6,5))
    plt.hist(err, bins=40, alpha=0.7)
    plt.title("Error Distribution (actual - estimated)")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    # error vs score
    plt.figure(figsize=(6,5))
    plt.scatter(score, err, alpha=0.4)
    plt.title("Error vs Score")
    plt.xlabel("Score")
    plt.ylabel("Error")
    plt.grid(True)
    plt.show()


def compute_statistics_and_plots(df, n_bins=20):
    """
    Computes:
      - Pearson correlation
      - Spearman rank correlations
      - NDCG(score, actual)
      - NDCG(estimated, actual)

    Generates:
      - scatter
      - hexbin
      - KDE (raw)
      - KDE (centered)
      - Log-KDE + fitted log-normal PDF
      - Centered Log-KDE + fitted log-normal PDF
      - calibration curve
    """

    score = df["value"].values
    est = df["user_attrs_r_hat"].values
    actual = df["user_attrs_actual_reward"].values
    ess = df["user_attrs_ess"].values

    err_hat = df["user_attrs_q_error"].values
    err = actual - est

    # metrics
    cor = compute_correlations(score, actual, est)
    ndcg_vals = compute_ndcg(score, actual, est)
    err_metrics = compute_error_metrics(actual, est)

    # plots
    plot_ranked_reward_curve(score, actual, est)
    plot_ess_heatmap_scatter(score, actual, ess)

    plot_kde([score, est, actual], ["Score", "Estimated", "Actual"])
    plot_centered_kde([score, est, actual], ["Score", "Estimated", "Actual"])
    plot_log_kde_with_lognormal_fit([score, est, actual], ["Score", "Estimated", "Actual"])
    plot_calibration_curve(score, actual)
    plot_error_plots(actual, est, score)

    plot_ess_heatmap_scatter(err_hat, err, ess)

    plot_error_hover(score, actual, ess, 
                     title="Interactive Score vs Actual Reward (ESS-colored)",
                    x_label="Score",
                    y_label="Actual Reward")
    
    plot_error_hover(err_hat, err, ess,
                    title="Interactive Estimated vs Actual Error (ESS-colored)",
                    x_label="Estimated Error",
                    y_label="Actual Error")

    # ===============================
    # Return metrics
    # ===============================
    print("Correlation Metrics:", cor)
    print("NDCG Metrics:", ndcg_vals)
    print("Error Metrics:", err_metrics)