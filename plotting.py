import matplotlib.pyplot as plt


def plot(pc, distance_matrix, reference_run=None, title='', ax=None, overlay=None, interactive=False, plotting=True, equal_axis=False, markers=False, clear=True, time_step=None, decay=None):
    labels = distance_matrix.member_names
    num_time_steps = distance_matrix.num_time_steps

    if overlay is not None and ax is None:
        interactive = False # Does not work together.
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if ax is not None:
        if clear:
            ax.clear()
        ax.set_xlabel('1st PC')
        ax.set_ylabel('2nd PC')
        ax.set_title(title)
        if equal_axis:
            ax.axes.set_aspect('equal')
        if overlay is not None:
            overlay(ax)
    else:
        if clear:
            plt.clf()
        ax = plt
        ax.title(title)
        ax.xlabel('1st PC')
        ax.ylabel('2nd PC')
        if equal_axis:
            ax.axis('equal')

    if markers:
        marker = '.'
        markersize = 5
    else:
        marker = None
        markersize = None

    last_offset = 0
    for i, offset in enumerate(num_time_steps):

        x = pc[0][last_offset:(last_offset + offset)]
        y = pc[1][last_offset:(last_offset + offset)]

        if len(y) == 1:
            ax.axhline(y[0], ls='-.', label=labels[i])
        else:
            if i == reference_run:
                ax.plot(x, y, label=labels[i], marker=marker, markersize=markersize, ls='--', zorder=1000)
                color = ax.get_lines()[-1].get_c() if ax != plt else plt.gca().get_lines()[-1].get_c()
                ax.scatter(x[0], y[0], marker='s', color=color)
                    #ax.scatter(x[-1], y[-1], marker='s', color=ax.get_lines()[-1].get_c())
            else:
                ax.plot(x, y, label=labels[i], marker=marker, markersize=markersize)

        # Draw time step marker.
        if time_step is not None:

            def get_pos(time):
                t = int(time)

                x_t0 = pc[0][last_offset + t]
                x_t1 = pc[0][last_offset + t + 1]

                y_t0 = pc[1][last_offset + t]
                y_t1 = pc[1][last_offset + t + 1]

                x_t = x_t0 + (x_t1 - x_t0) * (time - t)
                y_t = y_t0 + (y_t1 - y_t0) * (time - t)

                return x_t, y_t

            t = int(time_step)
            if t == offset-1:
                ax.scatter(pc[0][last_offset+t], pc[1][last_offset+t], marker='o', color='black', zorder=1000, s=15)
            elif t < offset-1:
                x_t, y_t = get_pos(time_step)
                ax.scatter(x_t, y_t, marker='o', color='black', zorder=1000, s=15)

            if decay is not None:
                half_k_size = 50
                lower_bound = max(0, t-half_k_size)-t
                upper_bound = min(offset-1, t+half_k_size+1)-t
                xs = []
                ys = []
                radii = []
                for i in range(lower_bound, upper_bound):
                    r = decay(i) * 15
                    if r < 1:
                        continue

                    time = time_step + i
                    x, y = get_pos(time)
                    xs.append(x)
                    ys.append(y)
                    radii.append(r)
                ax.scatter(xs, ys, marker='o', color='black', zorder=1000, s=radii)

        last_offset += offset

    if len(labels) < 20: # HACK
        ax.legend()

    if interactive:
        plt.draw_if_interactive()
    else:
        plt.savefig('results/' + title + '.pdf')
        if plotting:
            plt.show()


def plot_voronoi(pc, distance_matrix, reference_run, title=''):
    from scipy.spatial import Voronoi, voronoi_plot_2d

    reference_idx = distance_matrix.get_reference_idx(reference_run)
    p = pc[:,reference_idx[0]:reference_idx[1]].T
    voronoi = Voronoi(p)

    def overlay(ax):
        voronoi_plot_2d(voronoi, ax=ax, show_points=False, show_vertices=False)

    plot(pc, distance_matrix, title, reference_run, overlay)


def plot_heatmap(values, title='', plotting=True):
    plt.clf()
    plt.imshow(values)
    plt.title(title)
    plt.savefig('results/' + title + '.pdf')
    if plotting:
        plt.show()


def plot_bars(values, title='', plotting=True, labels=None):
    plt.clf()
    plt.bar(range(len(values)), values)
    plt.title(title)
    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.savefig('results/' + title + '.pdf')
    if plotting:
        plt.show()
