# This file contains all experiments to reproduce the results of the paper.
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

from distancematrix import DistanceMatrix, create_extended_distance_matrix
from plotting import plot
from simplification import simplify_reference_run, minimize_stress_along_reference_curve, \
    find_simple_curve
from techniques.classicalmds import ClassicalMdsEmbedding
from techniques.ours import smacof, OurEmbedding
from transformation import transform_distance_over_time, transform_1st_pc_over_time
from utils import distance_matrix_from_projection, generate_bump_curve
from weighting import create_weights_for_modified_distance_matrix, create_time_step_weights, \
    create_weights_for_extended_distance_matrix


def create_initial_embedding(distance_matrix, reference_run, decay, kernel_size):
    reference_idx = distance_matrix.get_reference_idx(reference_run)

    # Create the initial embedding.
    initial_projection = None  # distance_matrix.raw_data
    if initial_projection is None:
        # First, we use Classical MDS by Wickelmaier et al. to obtain a smooth (but not optimal) solution.
        high_dimensional_embedding = ClassicalMdsEmbedding(distance_matrix)
        high_dimensional_projection = high_dimensional_embedding.project()

        print(high_dimensional_embedding.get_eigenvalues())

        if distance_matrix.raw_data is not None:
            high_dimensional_projection = distance_matrix.raw_data

        # Then, instead of using this projection..
        # initial_projection = high_dimensional_projection[:2,:]

        # Use weights, same
        weights = create_weights_for_modified_distance_matrix(distance_matrix, reference_idx, 1)
        adjusted_weights = create_time_step_weights(distance_matrix, decay, k_size=kernel_size)
        weights *= adjusted_weights
        # weights = None

        # .. we further optimize it the initial embedding using SMACOF.
        initial_projection, _, _ = smacof(distance_matrix.matrix, high_dimensional_projection[:2, :].T, weights)
        initial_projection = initial_projection.T

        # TODO: investigate!
        # distance_matrix.matrix = distance_matrix_from_projection(initial_projection.T)

        # Plot the baseline.
        pc = transform_distance_over_time(distance_matrix, reference_run)
        plot(pc, distance_matrix, reference_run, title='Baseline highdimensional', plotting=False, interactive=False)
        pc = transform_1st_pc_over_time(initial_projection[0], distance_matrix, reference_run)
        plot(pc, distance_matrix, reference_run, title='Baseline 1st PC + time', plotting=False, interactive=False)
        pc = initial_projection
        plot(pc, distance_matrix, reference_run, title='Baseline 2D', plotting=False, interactive=False)
        plt.close()

    return initial_projection


def project(title, distance_matrix, reference_run, simplified_curve, initial_projection, kernel_size, decay=None, omega=None):
    reference_idx = distance_matrix.get_reference_idx(reference_run)
    target_distances = distance_matrix_from_projection(simplified_curve.T)
    dm = create_extended_distance_matrix(distance_matrix, reference_idx, target_distances)

    weights = None
    stress_tol = None
    if omega is not None:
        weights = create_weights_for_extended_distance_matrix(dm, reference_idx, omega)

        adjusted_weights = create_time_step_weights(dm, decay=decay, k_size=kernel_size)
        weights *= adjusted_weights

        # The following code, instead of the line above, will apply the weights to all
        # entries except for the lower right corner, i.e., the target run.
        # n_ref = reference_idx[1] - reference_idx[0]
        # weights[:-n_ref, :-n_ref] *= adjusted_weights[:-n_ref, :-n_ref]
        # weights[:-n_ref, :] *= adjusted_weights[:-n_ref, :]
        # weights[:, :-n_ref] *= adjusted_weights[:, :-n_ref]

    else:
        stress_tol = 0.1

    ip = np.concatenate((initial_projection, simplified_curve), axis=1)
    plot(ip, dm, reference_run, title=title + '_initial', plotting=False, interactive=False, markers=True)

    embedding = OurEmbedding(dm, dm, 2, None, weights, reference_run, ip, max_num_iterations, False, False, None, stress_tol)
    result = embedding.project()

    plot(result, distance_matrix, reference_run, title=title+'_result', plotting=False, interactive=False, markers=True)
    plot(result, dm, reference_run, title=title+'_with_target', plotting=False, interactive=False, markers=True)



def experiment_feature():

    reference_run = 1

    num_time_steps = [i*20 for i in range(1, 6)]
    feature_in_data = [False, True]

    sigmas = [0.5, 1, 2, 4]
    omegas = [1, 10, 100, 1000]

    for num_t, feature in itertools.product(num_time_steps, feature_in_data):

        # Setup distance matrix.
        if feature:
            radius = 0.2
        else:
            radius = 0
        distance_matrix = init_illustrative_dataset(num_t, radius)

        # We vary the kernel size and keep everything else fixed.
        for sigma, omega in itertools.product(sigmas, omegas):

            exp_name = f'exp_feature={feature}_t={num_t}_sigma={sigma}_omega={omega}'

            def gaussian(x):
                return np.exp(-x ** 2 / (2 * sigma ** 2))

            initial_projection = create_initial_embedding(distance_matrix, reference_run, decay=gaussian, kernel_size=None)

            if feature:
                # In case we have a feature, we try to straighten it.
                #reference_positions_2d = initial_projection[:, reference_idx[0]:reference_idx[1]]
                #simplified_curve = simplify_reference_run(reference_positions_2d,
                #                                          "approximated",
                #                                          epsilon=100,
                #                                          fix_start=True,
                #                                          fix_end=True,
                #                                          check_result=False
                #                                          )
                xs = np.linspace(-1, 1, num=num_t)
                ys = [-1] * num_t
                simplified_curve = np.column_stack((xs, ys)).T

            else:
                # In case we don't have a feature, we try to introduce it.
                feature_curve = generate_bump_curve(num_t, h=0.2, l=0.2).T
                #feature_curve[0, :] -= 1
                feature_curve[1, :] -= 1
                simplified_curve = feature_curve
                #simplified_curve = minimize_stress_along_reference_curve(reference_positions_2d, feature_curve, fix_start=True, fix_end=True, plotting_callback=None)

            project(exp_name, distance_matrix, reference_run, simplified_curve, initial_projection, kernel_size=None, decay=gaussian, omega=omega)



def experiment_all():
    datasets = list(sorted(os.listdir('datasets')))
    # for i, data_set in enumerate(datasets):
    #     print(f'{i}: {data_set}')

    # selected_dataset = int(input('Select dataset: '))
    selected_dataset = 9
    reference_run = 6

    # Run the application.
    path = 'datasets/' + datasets[selected_dataset]

    distance_matrix = DistanceMatrix().from_file(path)
    reference_idx = distance_matrix.get_reference_idx(reference_run)

    sigmas = [i*0.5 for i in range(1, 21, 5)]
    omegas = [1, 10, 100, 500, 1000]
    epsilons = [0.15, 0.4, 0.05]

    fix_start = True
    fix_end = True

    # We vary the kernel size and keep everything else fixed.
    for sigma, omega, epsilon in itertools.product(sigmas, omegas, epsilons):

        title = f'exp_all_sigma={sigma}_eps={epsilon}_omega={omega}'

        def gaussian(x):
            return np.exp(-x ** 2 / (2 * sigma ** 2))

        initial_projection = create_initial_embedding(distance_matrix, reference_run, decay=gaussian, kernel_size=None)
        reference_curve = initial_projection[:, reference_idx[0]:reference_idx[1]]

        smoothed_curve = find_simple_curve(reference_curve, 'smoothed', epsilon, fix_start=fix_start, fix_end=fix_end)
        simplified_curve = minimize_stress_along_reference_curve(reference_curve, smoothed_curve, fix_start=fix_start, fix_end=fix_end)

        project(title, distance_matrix, reference_run, simplified_curve, initial_projection, kernel_size=None, decay=gaussian, omega=omega)



if __name__ == '__main__':

    max_num_iterations = 150

    experiment_feature()
    #experiment_all()
