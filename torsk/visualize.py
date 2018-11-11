import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation


def animate_double_imshow(frames1, frames2,
                          time=None, vmin=None, vmax=None,
                          cmap_name="viridis", figsize=(12,4)):
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
    im1  = ax[0].imshow(frames1[0], animated=True, vmin=vmin, vmax=vmax,
                    cmap=plt.get_cmap(cmap_name))
    im2  = ax[1].imshow(frames2[0], animated=True, vmin=vmin, vmax=vmax,
                    cmap=plt.get_cmap(cmap_name))
    plt.colorbar(im2)
    text = ax[0].text(.5, 1.05, '', transform = ax[0].transAxes, va='center')
    if time is None:
        time = np.arange(len(frames1))
    def init():
        text.set_text("")
        im1.set_data(frames1[0])
        im2.set_data(frames2[0])
        return im1,im2,text
    def animate(i):
        text.set_text(str(time[i]))
        im1.set_data(frames1[i])
        im2.set_data(frames2[i])
        return im1,im2,text
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(frames1), interval=20, blit=True)
    return anim
