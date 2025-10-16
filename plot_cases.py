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
import re
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
    plot_single_task,
    probability_of_improvement,
    sample_efficiency_curves,
)

from matplotlib import pyplot as plt


from marl_eval.utils.data_processing_utils import (
    lower_case_inputs,
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


def plot_metric_from_json(directory, data, metric, step):
    results = {}

    task_name = None
    for env, tasks in data.items():
        for task, algorithms in tasks.items():
            print(task)
            match = re.search(r'_(\d+)coop', task)
            if match:
                num_coop = int(match.group(1))
            else:
                if task.endswith("adv_coop") or task.endswith("dis_coop"):
                    num_coop = 1
                elif task.endswith("both_coop"):
                    num_coop = 2
                else:
                    num_coop = 0
                    task_name = task

            for algorithm, seeds in algorithms.items():
                print(algorithm)
                if algorithm not in results:
                    results[algorithm] = {}

                if num_coop not in results[algorithm]:
                    results[algorithm][num_coop] = []

                for seed, steps in seeds.items():
                    if step in steps and metric in steps[step]:
                        results[algorithm][num_coop].extend(steps[step][metric])


    plt.figure(figsize=(10, 6))
    ordered_algorithms = sorted(results.keys(), key=lambda x: ("Cooperator" in x, x))
    ticks = set()
    for algorithm in ordered_algorithms:
        coop_data = results[algorithm]
        x_vals, y_means, y_errors = [], [], []
        for num_coop in sorted(coop_data.keys()):
            values = np.array(coop_data[num_coop])
            x_vals.append(num_coop)
            ticks.add(num_coop)
            y_means.append(values.mean())
            l, u = bootstrap_confidence_interval(values)
            y_errors.append([l, u])

        y_errors = np.array(y_errors).T

        ls = 'dashed'
        if 'cooperator' in algorithm.lower():
            ls = 'solid'

        c = 'black'
        if 'low-reward' in algorithm.lower() or "standard" in algorithm.lower():
            c = 'blue'
        elif 'high-reward' in algorithm.lower() or 'wide-zap' in algorithm.lower() or "spawn-biased" in algorithm.lower():
            c = 'red'


        algorithm = algorithm.replace(' Red', '').replace(' Blue', '')

        plt.plot(x_vals, y_means, label=algorithm, linestyle=ls, color=c)
        #plt.fill_between(x_vals, np.array(y_means) - y_errors[0], np.array(y_means) + y_errors[1], alpha=0.2, color=c)

    plt.xlabel("Number of Cooperators", fontsize='xx-large')
    plt.ylabel(metric_to_label(metric), fontsize='xx-large')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize='xx-large')
    plt.xticks(list(ticks), fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.grid(True, linestyle='dashed', linewidth=0.5)
    pdf_file_name = f"{directory}/result_{metric}_{step}_{task_name}.pdf"
    plt.savefig(pdf_file_name)
    plt.close()
    try:
        subprocess.run(["pdfcrop", pdf_file_name])
        os.remove(pdf_file_name)
    except Exception as e:
        print(e)

def plot_experiments(directory, json_files, metric_to_plot, step):
    if len(json_files) == 0:
        return
    raw_dict = load_and_merge_json_dicts(experiment_json_files)
    remove_uncommon_keys(raw_dict)
    absolute_metrics = find_absolute_metrics_keys(raw_dict)

    if 'return' in absolute_metrics:
        role_map = {}
        calculate_and_add_metrics(raw_dict, role_map=role_map)
        absolute_metrics += ['return_std', 'return_coef_var', 'return_rel_mad', 'equality_norm'] + [f'return_{r}' for r in list(role_map.keys())]

    raw_dict = replace_absolute_metrics_with_last_avg(raw_dict)

    if metric_to_plot is None:
        metrics_to_plot = absolute_metrics
    else:
        if metric_to_plot not in absolute_metrics:
            return
        metrics_to_plot = [metric_to_plot]

    for metric in sorted(metrics_to_plot):
        print(f"Plotting {metric}")
        plot_metric_from_json(directory, raw_dict, metric, step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--folder", default=None, help="folder containing experiment results")
    parser.add_argument("-m", "--metric", default=None, help="metric to plot")
    parser.add_argument("-s", "--step", default='step_0', help="step to plot")
    args = vars(parser.parse_args())

    metric_to_plot = args['metric']
    env_name = "meltingpot"
    directory = Path(args['folder']).absolute()

    # SSD metrics
    experiment_json_files = []
    for file_path in directory.rglob('*ssd_players*.json'):
        experiment_json_files.append(file_path)

    plot_experiments(directory, experiment_json_files, metric_to_plot, args['step'])

    # Regular metrics
    experiment_json_files = []
    for file_path in directory.rglob('*.json'):

        if 'ssd_player' in str(file_path):
            continue

        experiment_json_files.append(file_path)

    plot_experiments(directory, experiment_json_files, metric_to_plot, args['step'])
