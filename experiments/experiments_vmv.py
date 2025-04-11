import itertools
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import init_toy_2D
from masking import create_time_step_mask, create_reference_only_mask
from plotting import plot, plot_bars
from stress import stress_ex
from techniques.ours import OurEmbedding
from transformation import transform_1st_pc_over_time, transform_n_pcs_over_time, \
    transform_distance_over_time
from utils import distance_matrix_from_projection
from weighting import create_weights_for_modified_distance_matrix, create_time_step_weights


def exp_test_all_parameters():
    ranges = {}
    ranges['param_m'] = [3, 6, 12, 24]
    ranges['param_t'] = [20, 40]
    ranges['param_s'] = [0.0, 2.0, 4.0, 8.0]
    ranges['param_a'] = [1.0]
    ranges['param_w'] = [i*math.pow(10, j) for j in range(3) for i in range(1, 10)] + [1000.0]
    ranges['param_k'] = [None]  # [None, 3, 5, 7, 9, 11]
    return ranges


def exp_test_omega_n_l():
    ranges = {}
    ranges['param_m'] = [10 * i for i in range(1, 10)]
    ranges['param_t'] = [10]
    ranges['param_s'] = [2.0, 8.0]
    ranges['param_a'] = [1.0]
    ranges['param_w'] = [None]
    ranges['param_k'] = [None]
    ranges['stress_tol'] = [1.1, 1.2, 1.3, 1.4]
    return ranges


def exp_print_illustrative_example():
    ranges = {}
    ranges['param_m'] = [3]
    ranges['param_t'] = [20]
    ranges['param_s'] = [0.0, 8.0]
    ranges['param_a'] = [1.0]
    ranges['param_w'] = [5.0, 50.0, 500.0]
    ranges['param_k'] = [None]
    return ranges


def exp_name_from_params(params, parameter_ranges):
    param_values = []
    for name in parameter_ranges.keys():
        if name in params:
            s = f'{name[-1]}={params[name]}'
            param_values.append(s)

    return ','.join(param_values)


# Labels for stress currently implements.
STRESS_FULL = 'full'
STRESS_ONLY_TIME = 'time'
STRESS_ONLY_REFERENCE = 'ref'
STRESS_ONLY_TIME_AND_REFERENCE = 'ref+time'
STRESS_ONLY_TARGET = 'target'

# Labels for projections currently implemented.
PROJECTION_2D = '2D'
PROJECTION_1D_TIME = '1D+time'
PROJECTION_1D_TIME_CORRECTED = '1D+time(ref)'
PROJECTION_LOW_DIM_DIST = 'ld(ref)'
PROJECTION_HIGH_DIM_DIST = 'hd(ref)'


def calculate_stresses(original_distance_matrix, target_distance_matrix, pc, reference_run, label):
    stresses = dict()

    original_distances = original_distance_matrix.matrix

    projected_distances = distance_matrix_from_projection(pc.T)

    stresses['label'] = label
    stresses[STRESS_FULL] = stress_ex(projected_distances, original_distances)

    mask = create_time_step_mask(original_distance_matrix)
    stresses[STRESS_ONLY_TIME] = stress_ex(projected_distances, original_distances, mask=mask)

    if reference_run is not None:
        reference_idx = original_distance_matrix.get_reference_idx(reference_run)
        stresses[STRESS_ONLY_REFERENCE] = stress_ex(projected_distances, original_distances, reference=reference_idx)
        stresses[STRESS_ONLY_TIME_AND_REFERENCE] = stress_ex(projected_distances, original_distances, reference=reference_idx, mask=mask)
        target_distances = target_distance_matrix.matrix
        stresses[STRESS_ONLY_TARGET] = stress_ex(projected_distances, target_distances, mask=create_reference_only_mask(original_distance_matrix, reference_idx))

    return pd.DataFrame(stresses, index=[label])


def gather_stats(pc, params, name='', plotting=True, verbose=True):
    print('#'*80)
    print(name)

    original_distance_matrix = params['original_distance_matrix']
    target_distance_matrix = params['target_distance_matrix']
    reference_run = params['reference_run']

    stats = []
    stats.append(calculate_stresses(original_distance_matrix, target_distance_matrix, pc, reference_run, PROJECTION_2D))

    # Plot 2D embedding.
    plot(pc, original_distance_matrix, reference_run, f'Exp {name} - 2D Embedding', plotting=plotting)

    # Plot 1D+time embedding.
    proj = transform_1st_pc_over_time(pc, original_distance_matrix, None)
    stats.append(calculate_stresses(original_distance_matrix, target_distance_matrix, proj, reference_run, PROJECTION_1D_TIME))
    plot(proj, original_distance_matrix, reference_run, f'Exp {name} - 1st PC + Time', plotting=plotting and verbose)

    # Plot 1D+time embedding (corrected for reference run).
    proj = transform_1st_pc_over_time(pc, original_distance_matrix, reference_run)
    stats.append(calculate_stresses(original_distance_matrix, target_distance_matrix, proj, reference_run, PROJECTION_1D_TIME_CORRECTED))
    plot(proj, original_distance_matrix, reference_run, f'Exp {name} - 1st PC + Time (corrected)', plotting=plotting)

    if reference_run is not None:
        # Plot Voronoi.
        # plot_voronoi(pc, distance_matrix, reference_run)

        # Plot distance to reference in embedding.
        proj = transform_n_pcs_over_time(pc, original_distance_matrix, reference_run)
        stats.append(calculate_stresses(original_distance_matrix, target_distance_matrix, proj, reference_run, PROJECTION_LOW_DIM_DIST))
        plot(proj, original_distance_matrix, reference_run, f'Exp {name} - Distance to Reference (embedded)', plotting=plotting and verbose)

        # Plot high dimensional distances to reference.
        proj = transform_distance_over_time(original_distance_matrix, reference_run)
        stats.append(calculate_stresses(original_distance_matrix, target_distance_matrix, proj, reference_run, PROJECTION_HIGH_DIM_DIST))
        plot(proj, original_distance_matrix, reference_run, f'Exp {name} - Distance to Reference (high dim)', plotting=plotting and verbose)

    result = pd.concat(stats)
#    print(result.to_string())

    return result


def setup_single_experiment(params):

    initial_projection = params['initial_projection']
    original_distance_matrix = params['original_distance_matrix']
    target_distance_matrix = params['target_distance_matrix']
    num_dimensions = params['num_dimensions']
    reference_run = params['reference_run']
    reference_idx = original_distance_matrix.get_reference_idx(reference_run)

    if params['param_w'] is None:
        distance_matrix = target_distance_matrix
        weights = None
    else:
        # Set up the distance matrix.
        distance_matrix = original_distance_matrix.copy()
        alpha = params.get('param_a', 1.0)
        distance_matrix.matrix = (1-alpha) * original_distance_matrix.matrix + alpha * target_distance_matrix.matrix

        # Overwrite the distance matrix.
        params['target_distance_matrix'] = distance_matrix

        # Set up weight matrix.
        weights = create_weights_for_modified_distance_matrix(distance_matrix, reference_idx, params.get('param_w', 1.0))

        # Set up time step kernel.
        k_size = params.get('param_k', None)
        if k_size is not None:
            adjusted_weights = create_time_step_weights(distance_matrix, k_size=k_size)
            weights *= adjusted_weights

    draw_iterations = params.get('draw_iterations', False)
    stress_tol = params.get('stress_tol', None)
    return OurEmbedding(original_distance_matrix, distance_matrix, num_dimensions, None, weights, reference_run, initial_projection, 100, draw_iterations, False, None, stress_tol)


def setup_weights_experiments(reference_run, parameter_ranges):

    experiments = []

    stress_tol_ranges = parameter_ranges.get('stress_tol', [None])

    for param_t, param_m, stress_tol in itertools.product(parameter_ranges['param_t'], parameter_ranges['param_m'], stress_tol_ranges):

        distance_matrix = init_toy_2D(num_time_steps=param_t, num_functions=param_m)
        initial_projection = distance_matrix.raw_data
        reference_idx = distance_matrix.get_reference_idx(reference_run)

        baseline_params = {'original_distance_matrix': distance_matrix, 'target_distance_matrix': distance_matrix, 'reference_run': reference_run, 'param_m': param_m, 'param_t': param_t}
        baseline = ('Classical MDS', OurEmbedding(distance_matrix, distance_matrix, 2, None, np.ones_like(distance_matrix.matrix), reference_run, initial_projection, 100, False), baseline_params)
        experiments.append(baseline) # TODO: for the baseline, we should also apply 1. ClassicalMDS and 2. SMACOF.

        target_distance_matrix = distance_matrix.copy()

        x_values = np.linspace(-1, 1, num=param_t)

        for param_s in parameter_ranges['param_s']:

            y_values = param_s * x_values ** 2
            simplified_curve = np.array([x_values, y_values]).T

            target_distance_matrix.matrix[reference_idx[0]:reference_idx[1], reference_idx[0]:reference_idx[1]] = distance_matrix_from_projection(simplified_curve)

            for param_a, param_w, param_k in itertools.product(parameter_ranges['param_a'], parameter_ranges['param_w'], parameter_ranges['param_k']):
                params = {}

                params['initial_projection'] = initial_projection
                params['original_distance_matrix'] = distance_matrix
                params['target_distance_matrix'] = target_distance_matrix
                params['reference_run'] = reference_run
                params['num_dimensions'] = 2
                params['stress_tol'] = stress_tol

                params['param_m'] = param_m
                params['param_t'] = param_t
                params['param_s'] = param_s
                params['param_a'] = param_a
                params['param_w'] = param_w
                params['param_k'] = param_k

                exp_name = exp_name_from_params(params, parameter_ranges)
                exp = (exp_name, setup_single_experiment(params), params)
                experiments.append( exp )

    return experiments


def execute_experiments(experiments, parameter_ranges=None, plotting=False, verbose=False):
    print('[Main] Executing experiments')
    all_stats = []
    timings = {}
    for (name, embedding, params) in experiments:
        start = time.perf_counter()
        pc = embedding.project()
        timings[name] = time.perf_counter() - start

        stats = gather_stats(pc, params, name, plotting=plotting, verbose=verbose)

        # Add parameters.
        if parameter_ranges is not None:
            for param in parameter_ranges.keys():
                if param in params:
                    stats[param] = params[param]

        # For finding omega automatically, we need to add its value and stats.
        if embedding.omega is not None:
            stress_tol = params.get('stress_tol', None)
            if stress_tol is not None:
                stats['stress_tol'] = stress_tol
            stats['omega'] = embedding.omega
            stats['n'] = sum(params['original_distance_matrix'].num_time_steps)
            stats['l'] = params['original_distance_matrix'].num_time_steps[params['reference_run']]

        stats['technique'] = name
        stats['timing'] = timings[name]
        all_stats.append(stats)

    stats = pd.concat(all_stats)
    stats.to_csv('results/stats.csv')
    # stats = pd.read_csv('results/stats.csv')

    plt.close()
    stresses = [STRESS_FULL, STRESS_ONLY_TIME, STRESS_ONLY_REFERENCE, STRESS_ONLY_TIME_AND_REFERENCE, STRESS_ONLY_TARGET]
    projections = [PROJECTION_2D, PROJECTION_1D_TIME_CORRECTED]
    for projection in projections:
        result = stats.loc[(stats['label'] == projection)]
        columns = [col for col in result.columns if col in stresses]
        result[columns].plot(kind='bar', logy=True, title=projection)
        plt.xticks(range(len(result['technique'])), result['technique'])
        plt.savefig(f'results/stats_{projection}.png')
        if plotting:
            plt.show()

    # Plot the time they took.
    plot_bars(timings.values(), labels=timings.keys(), title='timings', plotting=plotting)

    print('Done!')


if __name__ == '__main__':
    plt.rcParams['font.size'] = 16
    new_margins = {'left': 0.15, 'right': 0.9, 'bottom': 0.15, 'top': 0.9}
    for key, value in new_margins.items():
        plt.rcParams[f'figure.subplot.{key}'] = value

    #parametrization = exp_test_all_parameters()
    #parametrization = exp_print_illustrative_example()
    parametrization = exp_test_omega_n_l()

    experiments = setup_weights_experiments(1, parametrization)
    execute_experiments(experiments, parameter_ranges=parametrization)
