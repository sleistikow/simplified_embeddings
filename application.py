import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider, Button, CheckButtons

from distancematrix import DistanceMatrix, create_extended_distance_matrix, \
    create_modified_distance_matrix
from plotting import plot
from simplification import simplify_reference_run, find_simple_curve
from techniques.ours import smacof
from techniques.techniques import *
from utils import distance_matrix_from_projection
from weighting import create_weights_for_modified_distance_matrix, create_time_step_weights, \
    create_weights_for_extended_distance_matrix, wip_experiment_with_weights

# Global variables.
reference_run = 1 # Default selected reference run.
reference_idx = (0, 0) # Respective indices (just a shorthand).
iterations = None # Storage for all iteration steps.

num_time_steps = 100 # Temporal Resolution.
max_num_iterations = 150

initial_projection = None # The initial projection.
decay = None # The current temporal decay function.


def calculate_initial_embedding(distance_matrix, sigma):
    # TODO: non-extended matrix version does no longer work that way.
    global decay
    if sigma is not None:
        decay = lambda x: np.exp(-x ** 2 / (2 * sigma ** 2))
    else:
        decay = None

    # Create the initial embedding.
    initial_projection = distance_matrix.raw_data
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
        adjusted_weights = create_time_step_weights(distance_matrix, decay=decay)
        weights *= adjusted_weights
        # weights = None

        # .. we further optimize it the initial embedding using SMACOF.
        initial_projection, _, _ = smacof(distance_matrix.matrix, high_dimensional_projection[:2, :].T, weights)
        initial_projection = initial_projection.T

        # Plot the baseline.
        #pc = transform_distance_over_time(distance_matrix, reference_run)
        #plot(pc, distance_matrix, reference_run, title='Baseline highdimensional', plotting=False, interactive=False)
        #pc = transform_1st_pc_over_time(initial_projection[0], distance_matrix, reference_run)
        #plot(pc, distance_matrix, reference_run, title='Baseline 1st PC + time', plotting=False, interactive=False)
        #pc = initial_projection
        #plot(pc, distance_matrix, reference_run, title='Baseline 2D', plotting=False, interactive=False)
        #plt.close()

    # Cut down to 2 dimensions.
    initial_projection = initial_projection[:2, :]

    return initial_projection

def main(path):

    # Load the selected dataset.
    distance_matrix = DistanceMatrix().from_file(path)

    global reference_run, reference_idx
    reference_idx = distance_matrix.get_reference_idx(reference_run)

    # Setup plot layout.
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(122)
    plt.subplots_adjust(wspace=0.3)

    # Create the initial embedding.
    sigma = 0.5  # Default sigma, it will be set by the user.
    global initial_projection
    initial_projection = calculate_initial_embedding(distance_matrix, sigma)

    plot(initial_projection, distance_matrix, reference_run, title='Initial Embedding', ax=ax, interactive=True)

    reference_run_slider_ax = fig.add_subplot(16, 2, 1)
    reference_run_slider = Slider(reference_run_slider_ax, 'Reference Run', 0, len(distance_matrix.member_names) - 1, valinit=reference_run, valstep=1, visible=True)

    sigma_slider_ax = fig.add_subplot(16, 2, 3)
    sigma_slider = Slider(sigma_slider_ax, 'Sigma', 0.5, 20, valinit=sigma, valstep=0.5)

    radio_ax = fig.add_subplot(7, 14, 15)
    radio_ax.axis('off')
    radio_buttons = RadioButtons(radio_ax, ('approximated', 'smoothed', 'user'))

    # This does not longer work with the newest matplotlib version.
    #for circle, label in zip(radio_buttons.circles, radio_buttons.labels):
    #    circle.set_radius(0.05)
    #    circle.set_edgecolor('#000000')
    #    circle.set_linewidth(2)
    #    circle.set_fill(True)
    #    label.set_fontsize(12)

    tuning_state_labels = ['Fix start', 'Fix end', 'Auto omega', 'Use temporal kernel', 'Extend matrix']
    tuning_state_labels_displayed = tuning_state_labels # The rest is, for now, enabled by default.
    tuning_states = [True, True, False, True, True]
    tuning_ax = fig.add_subplot(7, 14, 19)
    tuning_ax.axis('off')
    tuning_buttons = CheckButtons(tuning_ax, tuning_state_labels_displayed, tuning_states)

    # Just to make the code more readable (What is tuning_state[4]?).
    def get_tuning_state(label):
        return tuning_states[tuning_state_labels.index(label)]

    epsilon_slider_ax = fig.add_subplot(16, 2, 9)
    epsilon_slider = Slider(epsilon_slider_ax, 'Epsilon', 0.0001, 0.5, valinit=0.01)

    # TODO: make some suggestion for omega!
    # num_time_steps_all = distance_matrix.get_num_dimensions()
    # num_time_steps_ref = distance_matrix.num_time_steps[reference_run]
    # rho = num_time_steps_ref / num_time_steps_all

    omega_slider_ax = fig.add_subplot(16, 2, 11, label='omega')
    omega_slider = Slider(omega_slider_ax, 'Omega', 1, 1000, valinit=1, valstep=1)
    omega_slider.set_active(True)
    omega_slider_ax.set_visible(True)

    stress_tolerance_slider_ax = fig.add_subplot(16, 2, 11, label='stress_tolerance')
    stress_tolerance_slider = Slider(stress_tolerance_slider_ax, 'Stress tolerance', 0.0, 1.0, valinit=0.1, valstep=0.05)
    stress_tolerance_slider.set_active(False)
    stress_tolerance_slider_ax.set_visible(False)

    button_ax = fig.add_subplot(10, 2, 9)
    button = Button(button_ax, 'Apply')

    time_ax = fig.add_subplot(10, 2, 11)
    time_slider = Slider(time_ax, 'Time Step', 0, max(distance_matrix.num_time_steps)-1, valinit=0, valstep=0.1)

    result_ax = fig.add_subplot(10, 2, 13)
    result_slider = Slider(result_ax, 'Iteration', 0, 1, valinit=0, valstep=1)
    result_slider.set_active(False)

    stress_plot_ax = fig.add_subplot(3, 2, 5)

    def plotting_callback(title='', interactive=True):
        plot(initial_projection, distance_matrix, reference_run, title=title, ax=ax, interactive=interactive, time_step=time_slider.val, decay=decay)

    def sigma_slider_changed(sigma):
        global initial_projection

        initial_projection = calculate_initial_embedding(distance_matrix, sigma)

        plotting_callback()
        fig.canvas.draw()

    def radio_button_clicked(label):
        if label == 'user':
            epsilon_slider.set_active(False)
        else:
            epsilon_slider.set_active(True)

        plotting_callback()
        fig.canvas.draw()

    def reference_slider_changed(value):
        global reference_run, reference_idx
        reference_run = int(value)
        reference_idx = distance_matrix.get_reference_idx(reference_run)
        ax.clear()
        plot(initial_projection, distance_matrix, reference_run, title='Initial Embedding', ax=ax, interactive=True)
        fig.canvas.draw()

    def epsilon_slider_changed(epsilon):
        global reference_idx
        reference_positions_2d = initial_projection[:, reference_idx[0]:reference_idx[1]]
        # TODO: do we really want to fix the start and end point both in the optimization and in the simplification?
        fix_start = tuning_states[0]
        fix_end = tuning_states[1]
        simple_curve = find_simple_curve(reference_positions_2d, radio_buttons.value_selected, epsilon, fix_start=fix_start, fix_end=fix_end)
        ax.clear()
        plotting_callback()
        ax.plot(simple_curve[0], simple_curve[1], label='Simplified')
        fig.canvas.draw()

    def button_clicked(event):
        global reference_run, reference_idx
        reference_positions_2d = initial_projection[:, reference_idx[0]:reference_idx[1]]
        #simplified_curve = reference_positions_2d # Uncomment this line if you want the original curve.
        simplified_curve = simplify_reference_run(reference_positions_2d, radio_buttons.value_selected, ax=ax, epsilon=epsilon_slider.val, fix_start=tuning_states[0], fix_end=tuning_states[1], plotting_callback=plotting_callback)
        #simplified_curve = minimize_stress_along_reference_curve(reference_positions_2d, loop_curve, fix_start=tuning_states[0], fix_end=tuning_states[1], plotting_callback=plotting_callback, ax=ax)
        #simplified_curve = loop_curve

        # Handle user abort.
        if simplified_curve is None:
            return

        button.set_active(False)
        result_slider.set_active(False)
        stress_plot_ax.clear()
        fig.canvas.draw()

        target_distances = distance_matrix_from_projection(simplified_curve.T)

        if not get_tuning_state('Extend matrix'):

            # This branch is deprecated (our first approach where we modifiy, (i.e. not extend) the distance matrix).

            target_distance_matrix = create_modified_distance_matrix(distance_matrix, reference_idx, target_distances)

            weights = None
            stress_tol = None
            if not get_tuning_state('Auto omega'):

                weights = create_weights_for_modified_distance_matrix(distance_matrix, reference_idx, omega_slider.val)

                if get_tuning_state('Use temporal kernel'):
                    adjusted_weights = create_time_step_weights(distance_matrix, decay=decay)
                    weights *= adjusted_weights

            else:
                stress_tol = stress_tolerance_slider.val

            embedding = OurEmbedding(distance_matrix, target_distance_matrix, 2, None, weights, reference_run, initial_projection, max_num_iterations, True, True, ax, stress_tol, decay)
            embedding.project()

        else:

            # This branch implements our current approach, extending the distance matrix.

            dm = create_extended_distance_matrix(distance_matrix, reference_idx, target_distances)

            weights = None
            stress_tol = None
            if not get_tuning_state('Auto omega'):
                weights = create_weights_for_extended_distance_matrix(dm, reference_idx, omega_slider.val)

                if get_tuning_state('Use temporal kernel'):
                    adjusted_weights = create_time_step_weights(dm, decay=decay)

                    adjusted_weights = wip_experiment_with_weights(distance_matrix, reference_idx, adjusted_weights)

                    weights *= adjusted_weights

            else:
                stress_tol = stress_tolerance_slider.val

            # In this case, we need to add the simplified curve to the initial projection.
            ip = np.concatenate((initial_projection, simplified_curve), axis=1)

            embedding = OurEmbedding(dm, dm, 2, None, weights, reference_run, ip, max_num_iterations, True, True, ax, stress_tol, decay)
            embedding.project()

        global iterations
        iterations = embedding.iterations

        num_iterations = len(iterations)
        result_slider.ax.set_xlim(0, num_iterations-1)
        result_slider.valmax = num_iterations-1
        result_slider.set_val(0)
        result_slider.set_active(True)

        weighted_stress  = [it[1] for it in iterations]
        reference_stress = [it[2] for it in iterations]
        classical_stress = [it[3] for it in iterations]

        stress_plot_ax.plot(weighted_stress, label='Weighted Stress')
        stress_plot_ax.plot(reference_stress, label='Reference Stress $\\sigma^{ref}_1$')
        stress_plot_ax.plot(classical_stress, label='Classical Stress')
        stress_plot_ax.set_xlabel('Iteration')
        stress_plot_ax.set_ylabel('$\\sigma_1$')
        stress_plot_ax.legend()

        button.set_active(True)
        fig.canvas.draw()

    def time_slider_changed(_):
        global iterations
        if iterations is not None and len(iterations) > 0:
            result_slider_changed(result_slider.val)
        else:
            plotting_callback()

    def result_slider_changed(value):
        global iterations, decay
        ax.clear()
        # When does the following happen?
        if value >= len(iterations):
            value = len(iterations)-1
        plot(iterations[int(value)][0], distance_matrix, reference_run, title='Result', ax=ax, interactive=True, time_step=time_slider.val, decay=decay)
        fig.canvas.draw()

    def tuning_callback(label):
        index = tuning_state_labels.index(label)
        tuning_states[index] = not tuning_states[index]
        if label == 'Auto omega':
            omega_slider.set_active(not tuning_states[index])
            omega_slider_ax.set_visible(not tuning_states[index])
            stress_tolerance_slider.set_active(tuning_states[index])
            stress_tolerance_slider_ax.set_visible(tuning_states[index])
        # When the temporary kernel is disabled, the sigma slider should be disabled as well.
        elif label == 'Use temporal kernel':
            use_sigma = tuning_states[index]
            sigma_slider.set_active(use_sigma)
            if not use_sigma:
                sigma_slider_changed(None)
        fig.canvas.draw()

    # Connect the event handlers.
    sigma_slider.on_changed(sigma_slider_changed)
    radio_buttons.on_clicked(radio_button_clicked)
    epsilon_slider.on_changed(epsilon_slider_changed)
    reference_run_slider.on_changed(reference_slider_changed)
    tuning_buttons.on_clicked(tuning_callback)
    button.on_clicked(button_clicked)
    time_slider.on_changed(time_slider_changed)
    result_slider.on_changed(result_slider_changed)

    # Show the plot
    plt.show()


if __name__ == '__main__':

    # Allow the user to select a dataset.
    print('Available datasets:')
    datasets = list(sorted(os.listdir('datasets')))
    for i, data_set in enumerate(datasets):
        print(f'{i}: {data_set}')

    selected_dataset = int(input('Select dataset: '))

    # Run the application.
    main('datasets/'+datasets[selected_dataset])
