from techniques.embedding import Embedding

from utils import euclidean_distance
import matplotlib.pyplot as plt
import numpy as np
import sympy
import math


class LagrangeMultiplierEmbedding(Embedding):

    def __init__(self, distance_matrix, num_dimensions, mask, reference_run):

        self.distance_matrix = distance_matrix.matrix
        self.num_dimensions = num_dimensions

        # Set up a distance equation for each pair.
        vars, distances = self.generate_distance_matrix(self.distance_matrix, self.num_dimensions)
        self.num_vars = len(vars)

        # Nullify non relevant distances.
        distances[mask != 1] = 0

        # Remove distances that are not relevant.
        d = distances[distances != 0].flatten()
        p = self.distance_matrix[distances != 0].flatten()

        # Define Lagrange function:

        # Define Stress function.
        lagrange = self.generate_stress_function(d, p)

        # Add constraints.
        constraints = self.generate_constraints(vars, reference_run, self.num_dimensions)
        for i, constraint in enumerate(constraints):
            l = sympy.symbols(f'l_{i}')
            vars.append(l)
            lagrange = sympy.Add(lagrange, -l*constraint)

        # Derive by every symbol.
        symbols = []
        equations = []
        for x in vars:
            f = sympy.diff(lagrange, x)
            if f == 0:  # Remove trivial equations.
                continue
            symbols.append(x)
            eq = sympy.Eq(f, 0)
            equations.append(eq)

        # Set equations.
        self.symbols = symbols
        self.equations = equations

        # Print stats.
        print(f'Num symbols: {len(self.symbols)}')
        print(f'Num equations: {len(self.equations)}')

    def project(self):
        try:
            result = sympy.solve(self.equations, self.symbols, simplify=True)
            print(f'Found {len(result)} Results:')
            # for r in result:
            #    print(r)
        except Exception:
            print('Could not solve...')
            return None

        if len(result) == 0:
            print('No solution found!')
            return []

        # Take first result.
        result = result[0]

        # Cut off lagrangian multipliers.
        result = [float(result[i]) for i in range(self.num_vars)]

        # Reshape.
        result = np.array(result)
        num_members = int(self.num_vars / self.num_dimensions)
        result = np.reshape(result, (num_members, self.num_dimensions))

        return result.T

    @staticmethod
    def generate_distance_matrix(distance_matrix, num_dimensions):
        n = len(distance_matrix)

        # Setup one variable per time step and dimension.
        vars = [sympy.symbols(f'x_{d}_{i}') for i in range(n) for d in range(num_dimensions)]

        # Add distance functions.
        equations = np.full((n, n), fill_value=None, dtype=object)
        for i in range(n):
            for j in range(n):
                sum = sympy.Integer(0)
                for d in range(num_dimensions):
                    sum = sympy.Add(((vars[i*num_dimensions+d] - vars[j*num_dimensions+d]) ** 2), sum)

                func = sympy.sqrt(sum, evaluate=False)
                equations[i][j] = func

        return vars, equations

    @staticmethod
    def generate_stress_function(d, p):

        stress = sympy.Integer(0)

        for i in range(len(d)):
            stress = sympy.Add((d[i] - p[i])**2, stress)

        # We intentionally ignore dividing by the constant here.

        # Taking the square root might not be necessary...
        #stress = sympy.sqrt(stress)

        return stress

    @staticmethod
    def generate_constraints(vars, reference_run, num_dimensions, axis=1):
        constraints = []

        x_offset = 0
        for i in range(reference_run[0], reference_run[1]):
            constraints.append(vars[i * num_dimensions] - x_offset)
            constraints.append(vars[i*num_dimensions+axis])
            x_offset += 1

        return constraints


if __name__ == '__main__':

    k = 4

    # What sympy needs.
    sym_x = []
    sym_y = []

    p = []
    for i in range(k):
        alpha = 2 * math.pi / k * i
        x = math.cos(alpha)
        y = math.sin(alpha)
        p.append([x, y])
        sym_x.append(f'x_{i}')
        sym_y.append(f'y_{i}')

    distance_matrix = np.zeros((k, k))
    for i in range(len(p)):
        for j in range(len(p)):
            distance_matrix[i][j] = euclidean_distance(p[i], p[j])

    sym_x = sympy.symbols(' '.join(sym_x))
    sym_y = sympy.symbols(' '.join(sym_y))

    equations = []
    for i in range(k):
        for j in range(i+1, k):
            f = sympy.sqrt((sym_x[i]-sym_x[j])**2+(sym_y[i]-sym_y[j])**2)
            equations.append(sympy.Eq(f, distance_matrix[i][j]))

            # TODO: accepting an epsilon does not work!
            # f = abs(sympy.sqrt((sym_x[i]-sym_x[j])**2+(sym_y[i]-sym_y[j])**2) - distance_matrix[i][j])
            # equations.append(sympy.StrictLessThan(f, 0.01))

    equations.append(sympy.Eq(sym_x[0], 0))
    equations.append(sympy.Eq(sym_y[0], 0))

    symbols = []
    symbols.extend(sym_x)
    symbols.extend(sym_y)

    try:
        result = sympy.solve(equations, symbols)
    except Exception:
        print('Could not solve...')
        exit(1)

    if len(result) == 0:
        print('No solution found!')
        exit(0)

    print(result)

    # Iterate results.
    for idx in range(len(result)):

        xs = result[idx][:k]
        ys = result[idx][k:]

        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111)

        for i in range(k):
            for j in range(i, k):
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]])

        plt.show()
