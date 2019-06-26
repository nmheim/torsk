# coding: future_fstrings
import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from torsk.data.utils import normalize
import contextlib
import imageio

logger = logging.getLogger(__name__)


def plot_iteration(model, idx, inp, state):
    new_state = model.esn_cell.forward(inp, state)
    input_stack = model.esn_cell.input_map(inp)
    x_input = model.esn_cell.cat_input_map(input_stack)
    x_state = model.esn_cell.state_map(state)

    def vec_to_rect(vec):
        size = int(np.ceil(vec.shape[0]**.5))
        shape = (size, size)
        pad = np.zeros(size * size - vec.shape[0])
        rect = np.concatenate([vec, pad], axis=0).reshape(shape)
        return rect

    nr_plots_to_dims = {
        6: (2, 3),
        7: (2, 4),
        8: (2, 4),
        9: (3, 3),
        10: (2, 5),
        11: (3, 4),
        12: (3, 4),
        13: (3, 5),
        14: (3, 5),
        15: (3, 5),
        16: (4, 4),
        17: (3, 6),
        18: (3, 6)}
    nr_plots = len(input_stack) + 5
    size = int(np.ceil(nr_plots**.5))
    if nr_plots not in nr_plots_to_dims:
        raise ValueError("Too many input_map_specs to plot")

    height, width = nr_plots_to_dims[nr_plots]
    fig, ax = plt.subplots(height, width, figsize=(10, 10))
    ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    im = ax[0].imshow(inp)
    ax[0].set_title("image")
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(vec_to_rect(state))
    ax[1].set_title("state")
    plt.colorbar(im, ax=ax[1])

    for i in range(nr_plots - 5):
        x = input_stack[i]
        spec = model.esn_cell.input_map_specs[i]
        axi = ax[i+2]

        if spec["type"] == "random_weights":
            arr = vec_to_rect(x)
        else:
            arr = x.reshape(spec["dbg_size"])
        im = axi.imshow(arr)
        axi.set_title(f"Win(image)_{spec['type']} spec: {i}")
        plt.colorbar(im, ax=axi)

    im = ax[-3].imshow(vec_to_rect(x_state))
    ax[-3].set_title("W(state)")
    plt.colorbar(im, ax=ax[-3])

    im = ax[-2].imshow(vec_to_rect(x_state + x_input))
    ax[-2].set_title("W(state) + Win(image)")
    plt.colorbar(im, ax=ax[-2])

    im = ax[-1].imshow(vec_to_rect(new_state))
    ax[-1].set_title("tanh(W(state) + Win(image))")
    plt.colorbar(im, ax=ax[-1])

    fig.suptitle(f"Iteration {idx}")
    plt.show()


# Assumes data is already in range [0,1]
def to_byte(data,mask):
    return (data*255*(mask==False)).astype(np.uint8);


def write_video(filename,Ftxx,mask=None,fps=24,colormap=cm.viridis):   
    (nt,ny,nx) = Ftxx.shape;

    if(mask is None):
        mask = np.zeros(Ftxx.shape[1:],dtype=np.bool);

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)
        
    vmin, vmax = Ftxx.reshape(-1).min(), Ftxx.reshape(-1).max();
    writer = imageio.get_writer(
        filename, fps=fps, quality=10, macro_block_size=None)

    for i in range(nt):
        img_rgbaf = colormap(normalize(Ftxx[i],vmin,vmax));
        frame=to_byte(img_rgbaf[:,:,:3],mask[:,:,None]);
        writer.append_data(frame)
    writer.close()


def write_double_video(filename,Ftxx1,Ftxx2,mask=None,fps=24,colormap=cm.viridis):   
    assert(Ftxx1.shape == Ftxx2.shape);
    (nt,ny,nx) = Ftxx1.shape;
   
    Ftxx = np.empty((nt,ny,2*nx),dtype=Ftxx1.dtype);
    Ftxx[:,:,:nx] = Ftxx1;
    Ftxx[:,:,nx:] = Ftxx2;
    
    write_video(filename,Ftxx,mask,fps,colormap)


def animate_triple_imshow(frames1, frames2, frames3, frames4,
                          time=None, vmin=None, vmax=None,
                          cmap_name="inferno", figsize=(6, 6), title=None,
                          axes_labels=None):
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
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    ax = ax.flatten()
    if title is not None:
        fig.suptitle(title)

    im1 = ax[0].imshow(
        frames1[0], animated=True, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap_name))
    im2 = ax[1].imshow(
        frames2[0], animated=True, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap_name))
    im3 = ax[2].imshow(
        frames3[0], animated=True, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap_name))
    # trivial prediciton
    im4 = ax[3].imshow(
        frames4[0], animated=True, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap_name))

    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    plt.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04)
    plt.colorbar(im4, ax=ax[3], fraction=0.046, pad=0.04)
    text = ax[0].text(.5, 1.05, '', transform=ax[0].transAxes, va='center')
    if axes_labels is not None:
        for lbl, a in zip(axes_labels, ax):
            a.set_xlabel(lbl)
    plt.tight_layout()

    if time is None:
        time = np.arange(len(frames1))

    def init():
        text.set_text("")
        im1.set_data(frames1[0])
        im2.set_data(frames2[0])
        im3.set_data(frames3[0])
        im4.set_data(frames4[0])
        return im1, im2, im3, im4, text

    def animate(i):
        text.set_text(str(time[i]))
        im1.set_data(frames1[i])
        im2.set_data(frames2[i])
        im3.set_data(frames3[i])
        im4.set_data(frames4[i])
        return im1, im2, im3, text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(frames1),
        interval=20, blit=True)
    return anim

    

def animate_double_imshow(frames1, frames2,
                          time=None, vmin=None, vmax=None,
                          cmap_name="inferno", figsize=(12, 4), title=None):
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
    if title is not None:
        fig.suptitle(title)

    im1 = ax[0].imshow(
        frames1[0], animated=True, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap_name))
    im2 = ax[1].imshow(
        frames2[0], animated=True, vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap(cmap_name))
    # trivial prediciton
#    im3 = ax[2].imshow(
#        frames1[0], animated=True, vmin=vmin, vmax=vmax,
#        cmap=plt.get_cmap(cmap_name))

    plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
#    plt.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04)
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


def animate_imshow(frames, time=None, vmin=None, vmax=None,
                   cmap_name="inferno", figsize=(8, 5)):
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
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(frames[0], animated=True, vmin=vmin, vmax=vmax,
                   cmap=plt.get_cmap(cmap_name))
    plt.colorbar(im)
    text = ax.text(.5, 1.05, '', transform=ax.transAxes, va='center')

    if time is None:
        time = np.arange(len(frames))

    def init():
        text.set_text("")
        im.set_data(frames[0])
        return im, text

    def animate(i):
        text.set_text(str(time[i]))
        im.set_data(frames[i])
        return im, text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(frames), interval=20, blit=True)
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
