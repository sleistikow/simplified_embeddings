from tkinter import messagebox

from scipy.signal import savgol_filter
from skimage.measure import approximate_polygon
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from stress import stress_ex
from utils import distance_matrix_from_projection


def smooth_curve_from_control_points(control_points, num_points=50):
    x = [point[0] for point in control_points]
    y = [point[1] for point in control_points]

    kind = 'linear'
    if len(control_points) >= 4:
        kind = 'cubic'

    t = np.arange(len(control_points))
    t_new = np.linspace(min(t), max(t), num_points)
    curve_interpolator = interp1d(t, list(zip(x, y)), kind=kind, axis=0)
    curve = curve_interpolator(t_new)
    curve_x, curve_y = zip(*curve)

    return np.array([curve_x, curve_y])


def find_simple_curve(original_points, method, epsilon=0.005, fix_start=False, fix_end=False):
    if method == 'smoothed':
        window_length = int(epsilon*100)
        polyorder = 1
        # Ensure that savgol's conditions are met.
        if polyorder >= window_length:
            window_length = polyorder + 1
        if window_length % 2 == 0:
            window_length += 1

        points = savgol_filter(original_points, window_length=window_length, polyorder=polyorder)
        if fix_start:
            points = np.column_stack((original_points.T[0], points))
        if fix_end:
            points = np.column_stack((points, original_points.T[-1]))
        return points
    elif method == 'simplified':
        line = LineString(original_points.T).simplify(epsilon, preserve_topology=False)
        return np.asarray(line.xy)
    elif method == 'approximated':
        control_points = approximate_polygon(original_points.T, epsilon)
        return smooth_curve_from_control_points(control_points)
    elif method == 'fitted':
        points = original_points.T
        num_points = len(points)
        min_x = min(original_points[0])
        max_x = max(original_points[0])
        poly = Polynomial.fit(original_points[0], original_points[1], 2)
        x = np.linspace(min_x, max_x, num_points)
        y = poly(x)
        return np.array([x, y])
    elif method == 'user':
        control_points = []
        if fix_start:
            control_points.append(original_points.T[0])
        control_points.extend(plt.ginput(n=0, timeout=0))
        if fix_end:
            control_points.append(original_points.T[-1])
        return smooth_curve_from_control_points(control_points)
    elif method == 'none':
        return original_points
    else:
        print('Unknown method!')
        return original_points


def minimize_stress_along_reference_curve(original_points, simple_curve, fix_start=True, fix_end=True, ax=None, plotting_callback=None):
    """
    Given a reference curve and a simplified curve, this function finds a curve parametrization
    that minimizes the stress between the original points and the simplified curve.

    @param original_points: The original points.
    @param simple_curve: The simplified curve.
    @param fix_start: Whether the start point should be fixed, i.e., it is set to the start of the original curve.
    @param fix_end: Whether the end point should be fixed, i.e., it is set to the end of the original curve.
    @param ax: The axis to plot to (optional).
    @param plotting_callback: A callback that is called after each iteration (optional).
    """

    original_distances = distance_matrix_from_projection(original_points.T)
    points = simple_curve.T

    # Start and end point are not subject to optimization.
    p_start = original_points.T[0]
    p_end = original_points.T[-1]

    def segment_lengths_from_curve(points):
        differences = points[1:] - points[:-1]
        segment_lengths = np.linalg.norm(differences, axis=1)
        return segment_lengths

    original_segment_lengths = segment_lengths_from_curve(original_points.T)

    segment_length = segment_lengths_from_curve(points)
    arc_lengths = np.cumsum(segment_length)
    total_length = arc_lengths[-1]

    def piecewise_linear_curve(t):
        idx = np.searchsorted(arc_lengths, total_length*t)
        if idx == 0:
            return points[0]
        alpha = (total_length * t - arc_lengths[idx-1]) / (arc_lengths[idx] - arc_lengths[idx-1])
        pos = points[idx-1] * (1-alpha) + points[idx] * alpha
        return pos

    def curve(x):
        points = [piecewise_linear_curve(t) for t in x]
        if fix_start:
            points = [p_start] + points
        if fix_end:
            points = points + [p_end]
        return np.array(points)

    def stress_functional_full(x):
        projected_distances = distance_matrix_from_projection(curve(x))
        stress = stress_ex(projected_distances, original_distances, normalize=False)

        return stress

    def stress_functional_neighbors(x):
        projected_segment_length = segment_lengths_from_curve(curve(x))
        stress = stress_ex(projected_segment_length, original_segment_lengths, normalize=False)

        return stress

    n = original_distances.shape[0]

    # Define the initial guess for the layout.
    initial_layout = np.linspace(0, 1, n)
    if fix_start:
        initial_layout = initial_layout[1:]
    if fix_end:
        initial_layout = initial_layout[:-1]

    # Define the bounds for the optimization variables.
    bounds = [(0, 1) for _ in range(len(initial_layout))]

    # The following contraint guarantees that the points are ordered along the curve.
    def sequential_constraint(x):
        x = x + [1.0]
        cons = [x[i+1] - x[i] for i in range(len(x)-1)]
        return cons
    constraints = [{'type': 'ineq', 'fun': sequential_constraint}]

    # We plot the original curve in case nothing else is specified.
    if plotting_callback is None:
        def plotting_callback():
            if ax is None:
                plt.plot(original_points[0], original_points[1], label='Original')
            else:
                ax.plot(original_points[0], original_points[1], label='Original')

    def plot_state(x):
        points = np.array(curve(x)).T
        x = ax
        if x is None:
            plt.clf()
            x = plt
        else:
            ax.clear()

        plotting_callback()
        x.plot(simple_curve[0], simple_curve[1], label='Simplified')
        x.scatter(points[0], points[1], c='r', label='Current')
        x.legend()
        plt.draw_if_interactive()
        plt.pause(0.01)

    callback = plot_state
    # callback = None # Uncomment this line to disable plotting and greatly improve performance.

    # Choose the stress functional.
    # stress_functional = stress_functional_full
    stress_functional = stress_functional_neighbors

    res = minimize(stress_functional, initial_layout, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-5, callback=callback)
    # res.x = initial_layout # For debugging purposes, you could set the result to the initial, equidistant layout.

    # Extract the optimized layout.
    points = np.array(curve(res.x))

    return points.T


def simplify_reference_run(original_points, method, optimize=True, check_result=True, epsilon=0.005, fix_start=True, fix_end=False, ax=None, plotting_callback=None):
    """
    Given a specified method, this function simplifies the original curve.
    The available methods are:
      - smoothed: this will smooth the curve using a savgol filter
      - simplified : this method uses the Douglas-Peucker algorithm to simplify the curve
      - approximated: same as simplified, but using a different library
      - fitted: fits a 2nd degree curve
      - user: the user will draw a curve using control points
      - none: the original curve is returned

    Second, a curve parametrization is determined that minimizes stress between the original points
    and the simplified curve.

    If the user aborts the simplification process, None is returned.
    """

    simple_curve = find_simple_curve(original_points, method, epsilon, fix_start, fix_end)
    if plotting_callback:
        if ax is None:
            plt.clf()
            plotting_callback()
            plt.plot(simple_curve[0], simple_curve[1], label='Simplified')
            plt.legend()
            plt.savefig('results/simplified.pdf')
        else:
            ax.clear()
            plotting_callback()
            ax.plot(simple_curve[0], simple_curve[1], label='Simplified')
            ax.legend()
        plt.draw_if_interactive()
        plt.pause(0.01)

    if not np.array_equal(simple_curve, original_points) and optimize:
        simple_curve = minimize_stress_along_reference_curve(original_points, simple_curve, fix_start=fix_start, fix_end=fix_end, ax=ax, plotting_callback=plotting_callback)

    original_distances = distance_matrix_from_projection(original_points.T)
    simplified_distances = distance_matrix_from_projection(simple_curve.T)
    stress = stress_ex(simplified_distances, original_distances)
    if check_result and stress > 0.2: # -> 0.2 according to Kruskal's classification.
        choice = messagebox.askquestion('High Stress', f'The stress between the original and target curve is high ({stress:.2f}). Do you want to keep the result?')
        if choice == 'no':
            return None

    return simple_curve


if __name__ == '__main__':

    reference_positions = np.load('reference_positions.npy')

    # Only use the first two principal components.
    reference_positions_2d = reference_positions[:2, :60]

    plt.plot(reference_positions_2d[0], reference_positions_2d[1], label='Original')

    simple_curve_smoothed = find_simple_curve(reference_positions_2d, 'smoothed')
    simple_curve_simplified = find_simple_curve(reference_positions_2d, 'simplified')
    #simple_curve_approximated = find_simple_curve(reference_positions_2d, 'approximated')
    simple_curve_fitted = find_simple_curve(reference_positions_2d, 'fitted')
    simple_curve_user = find_simple_curve(reference_positions_2d, 'user')

    plt.close()

    plt.plot(simple_curve_smoothed[0], simple_curve_smoothed[1], label='Smoothed')
    plt.plot(simple_curve_simplified[0], simple_curve_simplified[1], label='Simplified')
    #plt.plot(simple_curve_approximated[0], simple_curve_approximated[1], label='Approximated')
    plt.plot(simple_curve_fitted[0], simple_curve_fitted[1], label='Fitted')
    plt.plot(simple_curve_user[0], simple_curve_user[1], label='User')
    plt.legend()
    plt.show()

    stress_optimized_curve = minimize_stress_along_reference_curve(reference_positions_2d, simple_curve_user, fix_start=False, fix_end=False)

    plt.clf()
    plt.plot(simple_curve_simplified[0], simple_curve_simplified[1], label='Simplified')
    plt.plot(stress_optimized_curve[0], stress_optimized_curve[1], label='Stress Optimized')
    plt.legend()
    plt.draw_if_interactive()
    plt.pause(100)
