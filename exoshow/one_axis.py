import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from exoshow.definitions import default_title


class OneAxes:
    """"

    Attributes
    ----------
    limits: stores the min and max of the limits (with logarithm applied if in that scale)
    """
    def __init__(self,
                 limits: pd.DataFrame | tuple[float, float],
                 label: str,
                 direction: str,
                 string: str = None,
                 logarithmic: bool | None = None,
                 inverted: bool = False,
                 verbose: bool = True,
                 color: tuple[str, str] = ('k', 'k'),
                 ):
        """

        Parameters
        ----------
        logarithmic:
           whether to use logarithmic scale. If None (default) it uses what is defined in default_title
        inverted
           whether to create a decreasing axes (not used when set explicitly)
        verbose:
           whether to print the final limits.
        limits: pd.DataFrame, tuple[float, float]
           explicit initial limits of the axes, or dataframe for auto-calculation from the value of label column
        label: str
           name of column in database
        direction: str
            either 'x' or 'y'
        string:
            human-readable version of label

        """
        self.label, self.string = self.set_label(label, string)
        self.inverted = inverted
        self.color = color

        if logarithmic is None:
            if label in default_title:
                logarithmic = default_title[label][0][1] == 'log'
            else:
                logarithmic = False
        self.logarithmic = logarithmic

        if direction.lower() not in ['x', 'y']:
            raise ValueError('direction must be either "x" or "y"')
        self.direction = direction.lower()

        if isinstance(limits, pd.DataFrame):
            self.limits = self.set_limits(limits[label])
        else:
            self.limits = self.set_limits(limits)

        if label in default_title and len(default_title[label]) == 2:
            self.opposite = default_title[label][1]
        else:
            self.opposite = None

        if verbose:
            if logarithmic:
                limits = 10 ** self.limits
            else:
                limits = self.limits
            print(f"{direction.upper()}-axis '{label}' ({self.string}) limits are [{limits[0]:.2g}, {limits[1]:.2g}]")

    def set_label(self, label, string=None):
        if string is None:
            if label in default_title:
                string = default_title[label][0][0]
            else:
                string = ' '.join([s.capitalize() for s in label.split('_')])
        self.string = string
        self.label = label

        return label, string

    def scale(self, value):
        if self.logarithmic:
            return np.log10(value)
        return value

    def set_logarithmic(self, use_log):
        self.logarithmic = use_log

    def set_inverted(self, use_invert):
        self.inverted = use_invert

    def set_limits(self,
                   data,
                   margin=None,
                   ) -> tuple[float, float]:
        """

        Parameters
        ----------
        margin:
          Margin for auto-compute... rcparam['margin'] if None
        data:
         either a pandas DataFrame, Series, list, tuple with the data, or a (min, max) tuple

        Returns
        -------
        min and max limits found

        """
        if margin is None:
            margin = matplotlib.rcParams[f"axes.{self.direction}margin"]

        if isinstance(data, pd.DataFrame):
            data = data[self.label]
        if isinstance(data, (pd.Series, list, tuple)):
            limits = np.array([min(data), max(data)])
            if self.logarithmic:
                limits = np.log10(limits)
            if self.inverted:
                limits = np.array(limits[1], limits[0])

            delta = limits[1] - limits[0]
            margin = delta * margin

            self.limits = np.array([limits[0] - margin, limits[1] + margin])
        else:
            self.limits = np.log10(data) if self.logarithmic else data

        return self.limits

    def in_axes(self,
                ax):

        if self.direction == 'x':
            ax.spines['bottom'].set_color(self.color[1])
            ax.spines['top'].set_color(self.color[0])
        elif self.direction == 'y':
            ax.spines['right'].set_color(self.color[1])
            ax.spines['left'].set_color(self.color[0])
        else:
            raise ValueError('direction must be either "x" or "y"')

        getattr(ax, f'{self.direction}axis').label.set_color(self.color[0])
        plt.setp(plt.getp(ax, f'{self.direction}ticklabels'), color=self.color[0])
        getattr(ax, f'set_{self.direction}label')(self.string)
        if self.logarithmic:
            self._set_log_ticks(ax, self.limits)
        getattr(ax, f'set_{self.direction}lim')(self.limits)

        if self.opposite is not None:
            # if configured opposite axis and 2D
            def forward(x) -> np.ndarray:
                if self.logarithmic:
                    x = 10**x
                return self.opposite[1](x)

            ax2 = getattr(ax, f"twin{'y' if self.direction == 'x' else 'x'}")()
            limits = forward(np.array(getattr(ax, f'get_{self.direction}lim')()))
            getattr(ax2, f'set_{self.direction}label')(self.opposite[0])
            if self.logarithmic:
                limits = np.log10(limits)
                self._set_log_ticks(ax2, limits)
            getattr(ax2, f'set_{self.direction}lim')(limits)

        return self

    def _set_log_ticks(self, ax, limits):
        limits = np.array(limits).astype(int) + np.array([0, 1])
        dtick = ((limits[1] - limits[0]) / 5 + 0.5).astype(int)
        if dtick == 0:
            dtick = 1
        tick_vals = limits[0] + dtick * np.arange(int((limits[1] - limits[0]) / dtick) + 1)
        ticks = tick_vals
        labels = [f"$10^{{{t}}}$" for t in tick_vals]
        getattr(ax, f"set_{self.direction}ticks")(ticks)
        getattr(ax, f"set_{self.direction}ticklabels")(labels)
        major_range = range(limits[0] - 1, limits[1])
        minor = np.log10(np.array([np.array([1, 2, 3, 4, 5, 6, 7, 8, 9.0]) * (10.0 ** major)
                                   for major in major_range]).flatten())
        getattr(ax, f"set_{self.direction}ticks")(minor, minor=True)
