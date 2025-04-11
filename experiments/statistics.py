import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from distancematrix import DistanceMatrix
from experiments.experiments_vmv import exp_test_all_parameters


def plot_stress_experiment(df, parameter_ranges, stress_types, selected_weight=None, param_m=None, param_t=None, param_s=None, param_a=None, param_k=None):
    figure = Figure()
    ax = figure.add_subplot(111)

    param_range_m = parameter_ranges['param_m']
    param_range_t = parameter_ranges['param_t']
    param_range_s = parameter_ranges['param_s']
    param_range_a = parameter_ranges['param_a']
    param_range_k = parameter_ranges['param_k']
    param_range_w = parameter_ranges['param_w']

    if param_m is not None:
        param_range_m = [param_range_m[param_m]]
    if param_t is not None:
        param_range_t = [param_range_t[param_t]]
    if param_s is not None:
        param_range_s = [param_range_s[param_s]]
    if param_a is not None:
        param_range_a = [param_range_a[param_a]]
    if param_k is not None:
        param_range_k = [param_range_k[param_k]]

    line_styles = ['-', '--', '-.', ':']

    for param_m in param_range_m:
        for param_t in param_range_t:
            for param_s in param_range_s:
                for param_a in param_range_a:
                    for param_k in param_range_k:

                        params = {}
                        params['param_m'] = param_m
                        params['param_t'] = param_t
                        params['param_s'] = param_s
                        # params['param_a'] = param_a
                        # params['param_k'] = param_k

                        #exp_name = exp_name_from_params(params, parameter_ranges)
                        #exp_name = f'$\\rho = {param_m}, t(x)={param_s}x^2$'
                        exp_name = f'$\\rho = {param_m}$'

                        stress_values_list = []
                        for i, stress_type in enumerate(stress_types):

                            stress_values = []
                            for param_w in param_range_w:

                                params['param_w'] = param_w

                                stress_value = df.loc[(df['label'] == '2D') &
                                                      (df['param_m'] == param_m) &
                                                      (df['param_t'] == param_t) &
                                                      (df['param_s'] == param_s) &
                                                      (df['param_a'] == param_a) &
                                                      (df['param_w'] == param_w),
                                                    stress_type]
                                stress_values.append(stress_value.values[0])

                            stress_symbol = '$\sigma_1$' if stress_type == 'full' else '$\sigma_1^{ref}$'
                            label = f'{exp_name}, {stress_symbol}'
                            color = None
                            if i > 0:
                                color = ax.get_lines()[-1].get_c()
                            ax.plot(param_range_w, stress_values, label=label, linestyle=line_styles[i], color=color, marker='.', markersize=5)
                            stress_values_list.append(stress_values)

                        #cumulative_stress = np.sum(np.array(stress_values_list), axis=0)
                        #ax.plot(param_range_w, cumulative_stress, label=f'Ours {exp_name} cumulative', linestyle=line_styles[i+1], color=color, marker='.', markersize=5)

    # Plot reference
    # stress_value = df.loc[(df['label'] == '2D') & (df['technique'] == 'Classical MDS'), 'full'].values[0]
    # ax.hlines(stress_value, 0, param_range_w[-1], label='2D MDS', linestyles='dotted', color='blue')

    # stress_value = df.loc[(df['label'] == '1D+time') & (df['technique'] == 'Classical MDS'), 'full'].values[0]
    # ax.hlines(stress_value, 0, weights[-1], label='1D+time', linestyles='dashed', color='black')

    if selected_weight is not None:
        ax.vlines(parameter_ranges['param_w'][selected_weight], ax.get_ylim()[0], ax.get_ylim()[1], colors='black')

    ax.legend()
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('$\sigma_1$')

    return figure
    #plt.show()


def plot_omega_n_l_experiment(df):
    figure = Figure()
    ax = figure.add_subplot(111)

    values = df['stress_tol'].dropna().unique()
    param_range_s = [2.0, 8.0]
    line_styles = ['-', '--']

    for v in values:
        for i, param_s in enumerate(param_range_s):
            n = df.loc[(df['stress_tol'] == v) & (df['param_s'] == param_s), 'n']
            l = df.loc[(df['stress_tol'] == v) & (df['param_s'] == param_s), 'l']
            rhos = n / l
            # rho = n**2 / l**2
            omegas = df.loc[(df['stress_tol'] == v) & (df['param_s'] == param_s), 'omega']

            rhos = list(rhos)
            omegas[omegas > 200] = 10
            omegas = list(omegas)

            color = None
            if i > 0:
                color = ax.get_lines()[-1].get_c()
            ax.plot(rhos, omegas, label=f'$\\tau={v}, \\alpha={param_s}$', linestyle=line_styles[i], marker='.', markersize=5, color=color)

    ax.set_xlabel('$n/l$')
    ax.set_ylabel('$\omega$')
    ax.legend()
    return figure


def plot_target_curves():
    figure = Figure()
    ax = figure.add_subplot(111)

    param_t = 20

    # line_styles = ['-', '--', '-.', ':']
    line_styles = ['-']*10

    distance_matrix = DistanceMatrix().init_toy_2D(param_t, 3)
    raw_data = distance_matrix.raw_data

    colors = [
        '#fecc5c',
        '#fd8d3c',
        '#f03b20',
        '#bd0026',
    ]

    x_values = np.linspace(-1, 1, num=param_t)

    reference = 1

    for i in range(3):
        y_values = raw_data[:, i*param_t:(i+1)*param_t][1]
        if i == reference:
            line_style = '--'
        else:
            line_style = '-'
        ax.plot(x_values, y_values, label=distance_matrix.member_names[i], linestyle=line_style, marker='.', markersize=5)

    param_range_s = [0, 2, 4, 8]
    for i, param_s in enumerate(param_range_s):
        y_values = param_s * x_values ** 2
        ax.plot(x_values, y_values, label=f'$t_{i+1}(x) = {param_s}x^2$', linestyle='-.', color=colors[i], marker='.', markersize=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    return figure


def plot_time_step_impact():
    plot_stress_experiment(pd.read_csv(stats_path), exp_test_all_parameters(), ['full', 'target'], param_t=0).savefig('results/paper_exp_time_impact_20.pdf')
    plot_stress_experiment(pd.read_csv(stats_path), exp_test_all_parameters(), ['full', 'target'], param_t=1).savefig('results/paper_exp_time_impact_40.pdf')


def plot_scaling_impact():
    plot_stress_experiment(pd.read_csv(stats_path), exp_test_all_parameters(), ['full', 'target'], param_t=0, param_s=0).savefig('results/paper_exp_scaling_impact_0.pdf')
    plot_stress_experiment(pd.read_csv(stats_path), exp_test_all_parameters(), ['full', 'target'], param_t=0, param_s=1).savefig('results/paper_exp_scaling_impact_2.pdf')
    plot_stress_experiment(pd.read_csv(stats_path), exp_test_all_parameters(), ['full', 'target'], param_t=0, param_s=2).savefig('results/paper_exp_scaling_impact_4.pdf')
    plot_stress_experiment(pd.read_csv(stats_path), exp_test_all_parameters(), ['full', 'target'], param_t=0, param_s=3).savefig('results/paper_exp_scaling_impact_8.pdf')


if __name__ == '__main__':

#    plt.rcParams['font.size'] = 16
#    new_margins = {'left': 0.15, 'right': 0.9, 'bottom': 0.15, 'top': 0.9}
#    for key, value in new_margins.items():
#        plt.rcParams[f'figure.subplot.{key}'] = value

    # stats_path = '/data/ownCloud/projects/linearizedembedding/results/results_paper/stats.csv'
    stats_path = 'results/stats_paper.csv'
    #plot_stress_experiment(pd.read_csv(stats_path), exp_test_all_parameters(), ['full', 'target']).savefig('results/experiment.pdf')
    #plot_time_step_impact()
    plot_scaling_impact()

    #plot_omega_n_l_experiment(pd.read_csv('results/omega_stats.csv')).savefig('results/optimal_omega.pdf')
    #plot_target_curves().savefig('results/target_curves.pdf')