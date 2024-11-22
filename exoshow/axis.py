import pandas as pd
from matplotlib import pyplot as plt

from .one_axis import OneAxes


class Axes:
    def __init__(self, xlabel, ylabel, df, color='k', title=""):
        """

        Parameters
        ----------
        xlabel: str
            dataframe column label to use in X-range
        ylabel: str
            dataframe column label to use in Y-range
        df: pd.DataFrame
             database from where to get initial values
        color: str, (str, str, str, str, str)
             either the color to use everywhere, or the color to use in tile, left, right, bottom, top axes.
        """
        self.fig = None
        self.axes = []
        self.title = title
        self._color = self.set_color(color)
        self.x_axes = OneAxes(df, xlabel, 'x',
                              color=(self._color[3], self._color[4]),
                              )
        self.y_axes = OneAxes(df, ylabel, 'y',
                              color=(self._color[1], self._color[2]),
                              )

    def set_xdata(self, label, string=None, logarithmic=None, inverted=False):
        self.set_data("x", label, string, logarithmic, inverted)

    def set_ydata(self, label, string=None, logarithmic=None, inverted=False):
        self.set_data("y", label, string, logarithmic, inverted)

    def set_data(self, direction, label, string=None, logarithmic=None, inverted=False):
        match direction:
            case 'x':
                axes = self.x_axes
            case 'y':
                axes = self.y_axes
            case _:
                raise ValueError("direction must be 'x' or 'y'")

        axes.set_label(label, string)
        axes.set_logarithmic(logarithmic)
        axes.set_inverted(inverted)

    def set_color(self, color):
        if isinstance(color, str):
            color = (color,)*5
        elif len(color) != 5:
            raise ValueError(f'Invalid color specification: {color}')

        self._color = color

        return color

    def fix_xlims(self, value, margin=None):
        self.x_axes.set_limits(value, margin=margin)

    def fix_ylims(self, value, margin=None):
        self.y_axes.set_limits(value, margin=margin)

    def clear(self):
        """
Start from empty figure
        """
        if self.fig is None:
            self.fig = plt.figure()
        self.fig.clf()

    def new_ax(self) -> plt.Axes:
        ax = self.fig.add_subplot(111)
        self.axes.append(ax)

        return ax

    def plot(self,
             ):
        self.clear()
        ax = self.new_ax()
        ax.axis('on')

        self.x_axes.in_axes(ax)
        self.y_axes.in_axes(ax)

        ax.title.set_color(self._color[0])
        ax.set_title(self.title)

        return ax
