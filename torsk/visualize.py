import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation


def animate_double_imshow(frames1, frames2,
                          time=None, vmin=None, vmax=None,
                          cmap_name="viridis", figsize=(12, 4)):
    def _blit_draw(self, artists, bg_cache):
        # Handles blitted drawing, which renders only the artists given instead
        # of the entire figure.
        updated_ax = []
        for a in artists:
            # If we haven't cached the background for this axes object, do
            # so now. This might not always be reliable, but it's an attempt
            # to automate the process.
            if a.axes not in bg_cache:
                # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
                # change here
                bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
            a.axes.draw_artist(a)
            updated_ax.append(a.axes)
        # After rendering all the needed artists, blit each axes individually.
        for ax in set(updated_ax):
            # and here
            # ax.figure.canvas.blit(ax.bbox)
            ax.figure.canvas.blit(ax.figure.bbox)
    matplotlib.animation.Animation._blit_draw = _blit_draw
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    im1 = ax[0].imshow(
        frames1[0], animated=True, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap_name))
    im2 = ax[1].imshow(
        frames2[0], animated=True, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap_name))
    plt.colorbar(im2)
    text = ax[0].text(.5, 1.05, '', transform=ax[0].transAxes, va='center')
    if time is None:
        time = np.arange(len(frames1))

    def init():
        text.set_text("")
        im1.set_data(frames1[0])
        im2.set_data(frames2[0])
        return im1, im2, text

    def animate(i):
        text.set_text(str(time[i]))
        im1.set_data(frames1[i])
        im2.set_data(frames2[i])
        return im1, im2, text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(frames1), interval=20, blit=True)
    return anim


def plot_mackey(predictions, labels, weights=None):

    def sort_output(output, error):
        sort = sorted(zip(output, error), key=lambda arg: arg[1].sum())
        sort = np.array(sort)
        sort_out, sort_err = sort[:, 0, :], sort[:, 1, :]
        return sort_out, sort_err

    error = np.abs(predictions - labels)
    predictions, _ = sort_output(predictions, error)
    labels, error = sort_output(labels, error)

    if weights is None:
        fig, ax = plt.subplots(3, 1)
    else:
        fig, ax = plt.subplots(4, 1)
        hist, bins = np.histogram(weights, bins=100)
        ax[3].plot(bins[:-1], hist, label=r"$W^{out}$ histogram")

    ax[0].plot(labels[0], label="Truth")
    ax[0].plot(predictions[0], label="Prediction")
    ax[1].plot(labels[-1])
    ax[1].plot(predictions[-1])

    mean, std = error.mean(axis=0), error.std(axis=0)
    ax[2].plot(mean, label=r"Mean Error $\mu$")
    ax[2].fill_between(
        np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.5,
        label=r"$\mu \pm \sigma$")

    ax[0].set_ylim(-0.1, 1.1)
    ax[1].set_ylim(-0.1, 1.1)
    ax[2].set_ylim(-0.1, 1.1)

    for a in ax:
        a.legend()

    plt.tight_layout()
    return fig, ax
