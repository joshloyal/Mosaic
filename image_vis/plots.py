import matplotlib.pyplot as plt


def deaxis(fig=None, ax=None):
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    if not isinstance(axes, list):
        axes = [axes]

    for ax_i in axes:
        ax_i.set_axis_off()


def pillow_to_matplotlib(image, **kwargs):
    f, ax = plt.subplots(**kwargs)
    fig = plt.imshow(image)
    deaxis(fig, ax)
    return fig
