#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
from typing import List
import argparse
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.figure import Figure
import colorcet as cc
import seaborn as sns

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rliable.plot_utils import _annotate_and_decorate_axis

from benchmarl.eval_results import load_and_merge_json_dicts, Plotting
from marl_eval.plotting_tools.plotting import (
    aggregate_scores,
    performance_profiles,
    probability_of_improvement,
    sample_efficiency_curves,
)

from matplotlib import pyplot as plt

from marl_eval.utils.data_processing_utils import (
    lower_case_inputs,
    get_and_aggregate_data_single_task
)

algs_to_compare = [
    ['IQL w/ SVO', 'IQL'],
    ['IQL w/ IA', 'IQL']
]


def metric_to_label(metric):
    m = metric.replace('agents_', '')
    if m == 'ext_return':
        return "Average Return"
    elif m == 'average_zapped_others':
        return "Other agents zapped"
    elif m == 'ext_return_equality':
        return "Equality"
    elif m == 'last_tsr_range':
        return '$e^{avg}$'
    elif m == 'average_age':
        return 'Average Age ($\hat{ \\tau }$)'
    elif m == 'proportion_own_coins':
        return 'Proportion of Own Coins'

    return m.replace('_', ' ').title()

def bootstrap_confidence_interval(dataset, confidence=0.95, iterations=10000, sample_size=1.0, statistic=np.mean):
    """
    Bootstrap the confidence intervals for a given sample of a population
    and a statistic.

    Args:
        dataset: A list of values, each a sample from an unknown population
        confidence: The confidence value (a float between 0 and 1.0)
        iterations: The number of iterations of resampling to perform
        sample_size: The sample size for each of the resampled (0 to 1.0
                     for 0 to 100% of the original data size)
        statistic: The statistic to use. This must be a function that accepts
                   a list of values and returns a single value.

    Returns:
        Returns the upper and lower values of the confidence interval.
    """
    dataset = np.array(dataset)  # Convert to numpy array if not already
    n_size = int(len(dataset) * sample_size)
    stats = np.zeros(iterations)

    for i in range(iterations):
        # Sample (with replacement) using numpy
        sample = np.random.choice(dataset, size=n_size, replace=True)
        # Calculate user-defined statistic and store it
        stats[i] = statistic(sample)

    stats = np.sort(stats)
    # Compute percentiles for the confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (confidence + (1 - confidence) / 2) * 100
    lval, uval = np.percentile(stats, [lower_percentile, upper_percentile])

    return lval, uval


def get_algorithm_styles(algorithms: List[str], palette="colorblind") -> Dict[str, Dict[str, str]]:
    """Returns a mapping of colors, linestyles and legends for each algorithm"""
    color_map = {}
    linestyle_map = {}
    legend_map = {algo: algo.title().replace('Fair&Localia', 'Fair&LocalIA').replace('Fair&Localsvo', 'Fair&LocalSVO').replace(' Ia', ' IA').replace(' Svo', ' SVO').replace('Iql', 'IQL').replace('W/', 'w/') for algo in algorithms}
    unmatched = []
    for algo in algorithms:
        # Color
        if "w/ svo" in algo:
            color_map[algo] = "green"
        elif "w/ ia" in algo:
            color_map[algo] = "orangered"
        elif "w/ fair&localsvo" in algo:
            color_map[algo] = "deepskyblue"
        elif "w/ fair&localia" in algo:
            color_map[algo] = "goldenrod"
        elif "iql" in algo:
            color_map[algo] = "blue"
        else:
            unmatched.append(algo)

        # Linestyle
        if "low-reward" in algo or "slow-move" in algo:
            linestyle_map[algo] = 'dotted'
        elif "high-reward" in algo or "wide-zap" in algo or "spawn-biased" in algo:
            linestyle_map[algo] = 'dashed'
        else:
            linestyle_map[algo] = 'solid'

    if len(set(color_map.values())) == 1:
        colors = sns.color_palette(palette, len(algorithms))
        for algo, color in zip(algorithms, colors):
            color_map[algo] = color

    return color_map, linestyle_map, legend_map


def aggregate_data_single_task_with_conf_intervals(
    processed_data: Dict[str, Any],
    metric_name: str,
    metrics_to_normalize: List[str],
    task_name: str,
    environment_name: str,
    bounds: str
) -> Dict[str, Any]:
    """Compute the 95% boostrapped CI over all independent \
        experiment runs at each evaluation step for a given \
        environment and task.

    Args:
        processed_data: Dictionary containing processed data.
        metric_name: Name of metric to aggregate.
        metrics_to_normalize: List of metrics to normalize.
        task_name: Name of task to aggregate.
        environment_name: Name of environment to aggregate.
    """

    mean_ci_lp_up = get_and_aggregate_data_single_task(processed_data, metric_name, metrics_to_normalize, task_name, environment_name)

    if metric_name in metrics_to_normalize:
        metric_to_find = f"mean_norm_{metric_name}"
    else:
        metric_to_find = f"mean_{metric_name}"

    # Get the data for the given metric and environment
    task_data = processed_data[environment_name][task_name]

    # Get the algorithm names, number of runs and total steps
    algorithms = list(task_data.keys())
    runs = list(task_data[algorithms[0]].keys())
    steps = list(task_data[algorithms[0]][runs[0]].keys())

    # Remove absolute metric from steps.
    steps = [step for step in steps if "absolute" not in step.lower()]

    for step in steps:
        # Loop over each algorithm
        for algorithm in algorithms:
            # Get the data for the given algorithm
            algorithm_data = task_data[algorithm]
            # Compute the 95% boostrapped CI for the given algorithm over all seeds at a given step
            run_total = []
            for run in runs:
                run_total.append(algorithm_data[run][step][metric_to_find])

            if "lp" not in mean_ci_lp_up[algorithm].keys() or "up" not in mean_ci_lp_up[algorithm].keys():
                mean_ci_lp_up[algorithm]["lp"] = []
                mean_ci_lp_up[algorithm]["up"] = []

            if len(run_total) > 1 and bounds == 'boostrapped CI':
                lp, up = bootstrap_confidence_interval(run_total)
                mean_ci_lp_up[algorithm]["lp"].append(lp)
                mean_ci_lp_up[algorithm]["up"].append(up)
            else:
                mean_ci_lp_up[algorithm]["lp"].append(np.min(run_total))
                mean_ci_lp_up[algorithm]["up"].append(np.max(run_total))

    return mean_ci_lp_up


def plot_single_task_curve(
    aggregated_data: Dict[str, Any],
    algorithms: list,
    colors: Optional[Dict] = None,
    color_palette: str = "colorblind",
    linestyles: Optional[Dict] = None,
    figsize: tuple = (7, 5),
    xlabel: str = "Number of Frames (in millions)",
    ylabel: str = "Aggregate Human Normalized Score",
    ax: Optional[Axes] = None,
    labelsize: str = "xx-large",
    ticklabelsize: str = "xx-large",
    legends: Optional[Dict] = None,
    run_times: Optional[Dict] = None,
    **kwargs: Any,
) -> Figure:

    extra_info = aggregated_data.pop("extra")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if algorithms is None:
        algorithms = list(aggregated_data.keys())

    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette))

    if linestyles is None:
        linestyles = {algorithm: "solid" for algorithm in algorithms}

    marker = kwargs.pop("marker", "o")
    linewidth = kwargs.pop("linewidth", 2)

    last_values = {}
    for algorithm in sorted(algorithms):
        print(f"\tPlotting {algorithm}")
        x_axis_len = len(aggregated_data[algorithm]["mean"])

        # Set x-axis values to match evaluation interval steps.
        x_axis_values = np.arange(x_axis_len) * extra_info["evaluation_interval"]

        if run_times is not None:
            x_axis_values = np.linspace(0, run_times[algorithm] / 60, x_axis_len)

        metric_values = np.array(aggregated_data[algorithm]["mean"])
        confidence_interval = np.array(aggregated_data[algorithm]["ci"])
        lower, upper = (
            np.array(aggregated_data[algorithm]["lp"]),
            np.array(aggregated_data[algorithm]["up"]),
        )

        print(f"\tLast Value: {metric_values[-1]:.3f} ({lower[-1]:.3f}, {upper[-1]:.3f})")
        last_values[algorithm] = metric_values[-1]

        if legends is not None:
            algorithm_name = legends[algorithm]
        else:
            algorithm_name = algorithm

        algorithm_name = algorithm_name.replace(' Red', '').replace(' Blue', '')

        ax.plot(
            x_axis_values,
            metric_values,
            color=colors[algorithm],
            marker=marker,
            linewidth=linewidth + 1 if 'fair' in algorithm.lower() else linewidth,
            label=algorithm_name,
            linestyle=linestyles[algorithm]
        )
        ax.fill_between(
            x_axis_values, y1=lower, y2=upper, color=colors[algorithm], alpha=0.1
        )

    if 'iql w/ fair&localia - strong (+1.5 per apple)' in algorithms and 'iql w/ fair&localia - weak (+0.5 per apple)' in algorithms:
        print(f"\t Difference from iql w/ fair&localia - strong (+1.5 per apple) to iql w/ fair&localia - weak (+0.5 per apple): {last_values['iql w/ fair&localia - strong (+1.5 per apple)'] - last_values['iql w/ fair&localia - weak (+0.5 per apple)']}")
        print(f"\t Ratio of iql w/ fair&localia - strong (+1.5 per apple) to iql w/ fair&localia - weak (+0.5 per apple): {last_values['iql w/ fair&localia - strong (+1.5 per apple)'] / last_values['iql w/ fair&localia - weak (+0.5 per apple)']}\n")
    elif 'iql w/ fair&localia - regular' in algorithms and '-iql w/ fair&localia - strong (wider zap beam)' in algorithms:
        print(f"\t Difference from -iql w/ fair&localia - strong (wider zap beam) to iql w/ fair&localia - regular: {last_values['-iql w/ fair&localia - strong (wider zap beam)'] - last_values['iql w/ fair&localia - regular']}")
        print(f"\t Ratio of -iql w/ fair&localia - strong (wider zap beam) to iql w/ fair&localia - regular: {last_values['-iql w/ fair&localia - strong (wider zap beam)'] / last_values['iql w/ fair&localia - regular']}\n")

    return _annotate_and_decorate_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        labelsize=labelsize,
        ticklabelsize=ticklabelsize,
        **kwargs,
    )

def plot_single(
    processed_data: Dict[str, Dict[str, Any]],
    environment_name: str,
    task_name: str,
    metric_name: str,
    metrics_to_normalize: List[str],
    xlabel: str = "Timesteps",
    run_times: Optional[Dict[str, float]] = None,
    color_palette=None,
    bounds="boostrapped CI"
) -> Figure:
    """Produces aggregated plot for a single task in an environment.

    Args:
        processed_data: Dictionary containing processed data.
        environment_name: Name of environment to produce plots for.
        task_name: Name of task to produce plots for.
        metric_name: Name of metric to produce plots for.
        metrics_to_normalize: List of metrics that are normalised.
        xlabel: Label for x-axis.
        run_times: Dictionary that maps each algorithm to the number of seconds it
            took to run. If None, then environment steps will be displayed.
    """

    metric_name, task_name, environment_name, metrics_to_normalize = lower_case_inputs(
        metric_name, task_name, environment_name, metrics_to_normalize
    )

    task_mean_ci_min_max_data = aggregate_data_single_task_with_conf_intervals( # aggregate with the addition of boostrapped confidence intervals
        processed_data=processed_data,
        environment_name=environment_name,
        metric_name=metric_name,
        task_name=task_name,
        metrics_to_normalize=metrics_to_normalize,
        bounds=bounds
    )

    if metric_name in metrics_to_normalize:
        ylabel = "Normalized " + " ".join(metric_name.split("_"))
    else:
        ylabel = " ".join(metric_name.split("_")).capitalize()

    # Upper case all algorithm names
    upper_algo_dict = {
        (algo.lower() if algo != "extra" else algo): value
        for algo, value in task_mean_ci_min_max_data.items()
    }
    task_mean_ci_min_max_data = upper_algo_dict
    algorithms = sorted(list(task_mean_ci_min_max_data.keys()))
    algorithms.remove("extra")

    if run_times is not None:
        run_times = {algo.lower(): value for algo, value in run_times.items()}
        xlabel = "Time (Minutes)"

    color_map, linestyle_map, legend_map = get_algorithm_styles(algorithms)

    fig = plot_single_task_curve(
        task_mean_ci_min_max_data,
        algorithms=algorithms,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=algorithms,
        figsize=(15, 8),
        colors=color_map,
        color_palette=color_palette,
        legends=legend_map,
        linestyles=linestyle_map,
        run_times=run_times,
        marker="",
    )

    return fig


# Function to find keys under 'absolute_metrics' in the JSON structure
def find_absolute_metrics_keys(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'absolute_metrics' and isinstance(value, dict):
                return list(value.keys())
            else:
                result = find_absolute_metrics_keys(value)
                if result:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = find_absolute_metrics_keys(item)
            if result:
                return result
    return None

def get_norm_equality(agent_returns):
    """
        Based on normalized Gini coefficient by
        Raffinetti, Emanuela, Elena Siletti, and Achille Vernizzi. "On the Gini coefficient normalization when attributes with negative values are considered." Statistical Methods & Applications 24, no. 3 (2015): 507-521.
    """

    N = agent_returns.shape[0]  # Number of agents
    if N == 1:
        return 1

    abs_differences = np.abs(agent_returns[:, None] - agent_returns[None, :]).sum()

    t_pos = np.asarray([max(0, r) for r in agent_returns]).sum()
    t_neg = abs(np.asarray([min(0, r) for r in agent_returns]).sum())
    delta_p = 2 * ((N - 1) / N**2) * (t_pos + t_neg)
    mu_py = (1/2) * delta_p
    equality = 1 - (abs_differences / (2 * mu_py * N**2 + 1e-5))
    return equality


def calculate_and_add_metrics(data, role_map=None):
    for env in data.keys():
        for task in data[env].keys():
            for alg in data[env][task].keys():
                print(f"There are {len(list(data[env][task][alg].keys()))} seeds for {alg} in {task}")
                for seed in data[env][task][alg].keys():
                    for s in data[env][task][alg][seed].keys():
                        if 'step' in s:
                            player_keys = sorted([k for k in data[env][task][alg][seed][s].keys() if 'player' in k])
                            N = len(player_keys)
                            num_episodes = len(data[env][task][alg][seed][s][player_keys[0]])
                            data[env][task][alg][seed][s]['return_std'] = [0] * num_episodes
                            data[env][task][alg][seed][s]['return_coef_var'] = [0] * num_episodes
                            data[env][task][alg][seed][s]['return_rel_mad'] = [0] * num_episodes
                            data[env][task][alg][seed][s]['equality_norm'] = [0] * num_episodes
                            for e in range(num_episodes):
                                returns = np.zeros(N)
                                for p in range(N):
                                    returns[p] = data[env][task][alg][seed][s][player_keys[p]][e]

                                data[env][task][alg][seed][s]['return_std'][e] = returns.std()
                                data[env][task][alg][seed][s]['return_coef_var'][e] = returns.std() / (returns.mean() + 1e-5)
                                data[env][task][alg][seed][s]['return_rel_mad'][e] = (1.0 / N) * np.abs(returns - returns.mean()).sum() / (returns.mean() + 1e-5)

                                data[env][task][alg][seed][s]['equality_norm'][e] = get_norm_equality(returns)

                            for role in role_map.keys():
                                data[env][task][alg][seed][s][f'return_{role}'] = [0] * num_episodes

                                for e in range(num_episodes):
                                    returns = []
                                    for p in role_map[role]:
                                        returns.append( data[env][task][alg][seed][s][f'{p}_return'][e] )

                                    data[env][task][alg][seed][s][f'return_{role}'][e] = np.asarray(returns).mean()

def replace_absolute_metrics_with_last_avg(data):
    """
    Replaces absolute_metrics with the average of the last `num_last_evals` evaluation episodes.

    Args:
        data (dict): The raw dictionary containing experiment results.
        num_last_evals (int): Number of last evaluation episodes to average over.

    Returns:
        dict: Updated data dictionary with modified absolute_metrics.
    """
    for env in data.keys():
        for task in data[env].keys():
            for alg in data[env][task].keys():
                for seed in data[env][task][alg].keys():
                    last_values = {}  # Store last few evaluation values for averaging

                    # Iterate through all steps to find the last ones
                    step_keys = sorted([s for s in data[env][task][alg][seed].keys() if "step" in s],
                                       key=lambda x: int(x.split("_")[-1]))

                    last_step = step_keys[-1]

                    for metric_name, values in data[env][task][alg][seed][last_step].items():
                        if metric_name == "step_count":
                            continue  # Skip step counter

                        if metric_name not in last_values:
                            last_values[metric_name] = []

                        last_values[metric_name].extend(values)  # Collect all last N values

                    # Compute averages and store in absolute_metrics
                    if "absolute_metrics" not in data[env][task][alg][seed]:
                        data[env][task][alg][seed]["absolute_metrics"] = {}

                    for metric_name, values in last_values.items():
                        if values:
                            data[env][task][alg][seed]["absolute_metrics"][metric_name] = [np.mean(values)]

    return data

def remove_uncommon_keys(data):
    print("Checking for uncommon keys...")
    task_key = list(data["meltingpot"].keys())[0]  # Extract the task key

    metric_sets = []
    for alg, alg_data in data["meltingpot"][task_key].items():
        seed_key = list(alg_data.keys())[0]  # Extract the seed key
        metric_sets.append(set(alg_data[seed_key]['absolute_metrics'].keys()))

    if metric_sets:
        common_metrics_set = set.intersection(*metric_sets)

        for alg, alg_data in data["meltingpot"][task_key].items():
            seed_key = list(alg_data.keys())[0]  # Extract the seed key

            keys_to_remove = set(alg_data[seed_key]["absolute_metrics"].keys()) - common_metrics_set
            for step, values in alg_data[seed_key].items():
                for key in keys_to_remove:
                    if step == "absolute_metrics":
                        print(f"\tRemoving {key} from {alg}")
                    del values[key]


def plot_experiments(json_files, metric_to_plot, env_name, bounds):
    if len(json_files) == 0:
        return
    raw_dict = load_and_merge_json_dicts(experiment_json_files)
    remove_uncommon_keys(raw_dict)
    absolute_metrics = find_absolute_metrics_keys(raw_dict)

    raw_dict = replace_absolute_metrics_with_last_avg(raw_dict)

    if metric_to_plot is None:
        metrics_to_plot = absolute_metrics
    else:
        if metric_to_plot not in absolute_metrics:
            return
        metrics_to_plot = [metric_to_plot]

    for metric in sorted(metrics_to_plot):
        Plotting.METRIC_TO_PLOT = metric
        Plotting.METRICS_TO_NORMALIZE = []
        processed_data = Plotting.process_data(raw_dict)
        (
            environment_comparison_matrix,
            sample_efficiency_matrix,
        ) = Plotting.create_matrices(processed_data, env_name=env_name) # makes changes on the processed_data

        for task_name in raw_dict[env_name].keys():
            keys_to_compare = []
            for pair in algs_to_compare:
                if pair[0] in raw_dict[env_name][task_name].keys() and pair[1] in raw_dict[env_name][task_name].keys():
                    keys_to_compare.append(pair)

            fig, ax = plt.subplots(figsize=(15, 8))  # Create a figure and axis

            print(f"Plotting {metric}")
            plot_single( # get_and_aggregate_data_single_task method (called within) expects every algorithm to have the same run names (ex: seed_0, seed_1)
                processed_data=processed_data,
                environment_name=env_name,
                task_name=task_name,
                metric_name=metric,
                metrics_to_normalize=[],
                color_palette=cc.glasbey_category10,
                bounds=bounds
            )

            if args['ymin'] and args['ymax']:
                plt.ylim(float(args['ymin']), float(args['ymax']))
            plt.xlabel("Timesteps", fontsize=24)
            plt.ylabel(metric_to_label(metric), fontsize=24)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.gca().xaxis.get_offset_text().set_fontsize(14)
            pdf_file_name = f"{directory}/result_{metric}_{task_name}.pdf"
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=2, fontsize=22)
            plt.savefig(pdf_file_name, bbox_inches='tight')
            plt.close()
            try:
                subprocess.run(["pdfcrop", pdf_file_name])
                os.remove(pdf_file_name)
            except Exception as e:
                print(e)


def plot_multiple_metrics_experiments(
    json_files: List[Path],
    metrics: List[str],
    env_name: str,
    bounds: str,
) -> None:
    """
    Process experiments like `plot_experiments`, but plot multiple metrics side by side
    with a single legend shared above all subplots.

    Args:
        json_files: list of experiment json files (same input as plot_experiments)
        metrics: list of metric names to plot
        env_name: environment name, e.g., "meltingpot"
        bounds: "boostrapped CI" or min/max (same as plot_experiments)
    """
    if len(json_files) == 0:
        return

    # Load and prepare raw data (same flow as plot_experiments)
    raw_dict = load_and_merge_json_dicts(json_files)
    remove_uncommon_keys(raw_dict)
    absolute_metrics = find_absolute_metrics_keys(raw_dict)
    raw_dict = replace_absolute_metrics_with_last_avg(raw_dict)

    # Filter requested metrics to ones that actually exist
    metrics_to_plot = []
    for m in metrics:
        if m in absolute_metrics:
            metrics_to_plot.append(m)
        else:
            print(f"[warn] Requested metric '{m}' not found in absolute_metrics; skipping.")

    if not metrics_to_plot:
        print("[warn] No valid metrics to plot after filtering.")
        return

    # One figure per task; each subplot is a metric
    for task_name in raw_dict[env_name].keys():
        print(f"Plotting SUBPLOTS for task '{task_name}' with metrics: {metrics_to_plot}")

        fig, axes = plt.subplots(
            1, len(metrics_to_plot),
            figsize=(8 * len(metrics_to_plot), 6),
            sharex=True
        )
        if len(metrics_to_plot) == 1:
            axes = [axes]  # make iterable

        all_handles, all_labels = [], []

        for ax, metric in zip(axes, metrics_to_plot):
            Plotting.METRIC_TO_PLOT = metric
            Plotting.METRICS_TO_NORMALIZE = []

            processed_data = Plotting.process_data(raw_dict)
            _environment_comparison_matrix, _sample_efficiency_matrix = Plotting.create_matrices(
                processed_data, env_name=env_name
            )

            metric_lc, task_lc, env_lc, metrics_norm_lc = lower_case_inputs(
                metric, task_name, env_name, []
            )
            agg = aggregate_data_single_task_with_conf_intervals(
                processed_data=processed_data,
                metric_name=metric_lc,
                metrics_to_normalize=metrics_norm_lc,
                task_name=task_lc,
                environment_name=env_lc,
                bounds=bounds
            )

            # Normalize algo names
            agg = {(k.lower() if k != "extra" else k): v for k, v in agg.items()}
            algorithms = sorted([k for k in agg.keys() if k != "extra"])

            # Styling per algorithm
            color_map, linestyle_map, legend_map = get_algorithm_styles(algorithms)
            legends = legend_map  # use as-is

            # Plot in the subplot
            handles = plot_single_task_curve(
                aggregated_data=agg,
                algorithms=algorithms,
                colors=color_map,
                color_palette=cc.glasbey_category10,
                linestyles=linestyle_map,
                figsize=None,
                xlabel="Timesteps",
                ylabel='',
                ax=ax,
                legends=legends,
                marker="",
            )

            # Collect handles for shared legend
            h, l = ax.get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)

            ax.set_title(metric_to_label(metric), fontsize=22)
            ax.tick_params(axis='both', labelsize=22)
            ax.set_xlabel("Timesteps", fontsize=20)
            ax.xaxis.get_offset_text().set_fontsize(14)


        # Remove legends from individual subplots
        for ax in axes:
            ax.legend().remove()

        # Shared legend above all subplots
        fig.legend(
            all_handles[:len(all_labels)//len(metrics_to_plot)],  # deduplicate
            all_labels[:len(all_labels)//len(metrics_to_plot)],
            loc='upper center',
            bbox_to_anchor=(0.5, 1.11),
            ncol=3,
            fontsize=22
        )

        fig.tight_layout(rect=[0, 0, 1, 0.9])

        pdf_file_name = f"{directory}/result_{'+'.join(metrics_to_plot)}_{task_name}.pdf"
        plt.savefig(pdf_file_name, bbox_inches='tight')
        plt.close()

        try:
            subprocess.run(["pdfcrop", pdf_file_name])
            os.remove(pdf_file_name)
        except Exception as e:
            print(e)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--folder", default=None, help="folder containing experiment results")
    parser.add_argument("-m", "--metric", default=None, help="metric to plot")
    parser.add_argument("-ymin", "--ymin", default=None, help="minimum value of y axis")
    parser.add_argument("-ymax", "--ymax", default=None, help="maximum value of y axis")
    parser.add_argument("-b", "--bounds", default='boostrapped CI', help="what to draw as error bounds")
    args = vars(parser.parse_args())

    metric_to_plot = args['metric']
    env_name = "meltingpot"
    directory = Path(args['folder']).absolute()

    # SSD metrics
    experiment_json_files = []
    for file_path in directory.rglob('*ssd_players*.json'):
        experiment_json_files.append(file_path)

    plot_experiments(experiment_json_files, metric_to_plot, env_name, args['bounds'])
    """
    # Example: overlay three metrics together
    #list_of_metrics = ["ext_return", "proportion_own_coins"]
    list_of_metrics = ["ext_return", "average_zapped_others", "peace", "sustainability"]
    plot_multiple_metrics_experiments(
        json_files=experiment_json_files,
        metrics=list_of_metrics,
        env_name=env_name,
        bounds=args['bounds'],
    )
    """
    # Regular metrics
    experiment_json_files = []
    for file_path in directory.rglob('*.json'):

        if 'ssd_player' in str(file_path):
            continue

        experiment_json_files.append(file_path)

    plot_experiments(experiment_json_files, metric_to_plot, env_name, args['bounds'])
