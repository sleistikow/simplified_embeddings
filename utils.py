import numpy as np

from stress import stress_ex


def euclidean_distance(p, q):
    return np.linalg.norm(np.asarray(p) - np.asarray(q))


def distance_matrix_from_projection(p):
    p = np.asarray(p)
    n = len(p)
    d = np.zeros((n, n))
    for i in range(n):
        d[i] = np.linalg.norm(p[i] - p, axis=1)

    return d


def get_kruskal_stress_class(normalized_stress):
    if normalized_stress < 0.025:
        return 0
    elif normalized_stress < 0.05:
        return 1
    elif normalized_stress < 0.1:
        return 2
    elif normalized_stress < 0.2:
        return 3
    else:
        return 4


def generate_target_positions(distance_matrix, reference, num_dimensions = None):
    if reference is None:
        return None

    if num_dimensions is None:
        num_dimensions = len(distance_matrix)

    # Gather distances between nodes.
    distances = [0]
    for i in range(reference[0] + 1, reference[1]):
        distances.append(distances[-1] + distance_matrix[i - 1, i])

    # Define new positions on the horizontal axis.
    target_positions = [[d] + [0 for _ in range(num_dimensions-1)] for d in distances]
    return np.array(target_positions)


def generate_loop_curve(num_timesteps, r=0.2):
    num_points_segment = num_timesteps // 5

    segment1 = np.column_stack((np.linspace(0, 0.5 - r / 2, num_points_segment), np.zeros(num_points_segment)))
    segment2 = np.column_stack((0.5 - r / 2 + r * np.cos(np.linspace(-np.pi / 2, 0, num_points_segment)), r + r * np.sin(np.linspace(-np.pi / 2, 0, num_points_segment))))
    segment3 = np.column_stack((0.5 + r / 2 * np.cos(np.linspace(0, np.pi, num_points_segment)), r + r / 2 * np.sin(np.linspace(0, np.pi, num_points_segment))))
    segment4 = np.column_stack((0.5 + r / 2 + r * np.cos(np.linspace(np.pi, 3 * np.pi / 2, num_points_segment)), r + r * np.sin(np.linspace(np.pi, 3 * np.pi / 2, num_points_segment))))
    segment5 = np.column_stack((np.linspace(0.5 + r / 2, 1, num_points_segment), np.zeros(num_points_segment)))

    curve = np.vstack((segment1, segment2, segment3, segment4, segment5))

    return curve


def generate_bump_curve(num_time_steps, h=0.01, l=0.1):
    xs = np.linspace(-1, 1, num=num_time_steps)
    ys = [0] * num_time_steps
    curve = np.column_stack((xs, ys))

    # Define the indices for the second and third segments.
    segment_start = int(num_time_steps * (0.5 - l / 2))
    segment_end = int(num_time_steps * (0.5 + l / 2))

    # Calculate the sine wave values for the desired range (0 to π).
    sine_wave = h*(1+np.sin(np.linspace(-np.pi/2, 3*np.pi/2, segment_end - segment_start)))

    # Replace the y-coordinates of the second and third segments.
    curve[segment_start:segment_end, 1] = sine_wave

    return curve


# Test suite.
if __name__ == '__main__':
    p = [0, 0, 0]
    q = [1, 1, 1]
    r = [1, 1, 0]

    distance = euclidean_distance(p, q)
    print(f'Euclidean distance={distance}')

    projection = np.asarray([p, q])
    mat_a = distance_matrix_from_projection(projection)
    projection = np.asarray([p, r])
    mat_b = distance_matrix_from_projection(projection)

    stress = stress_ex(mat_a, mat_b)
    print(f'Stress={stress}')
