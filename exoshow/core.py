# properties initialization
from pathlib import Path

from matplotlib import pyplot as plt

from exoshow import db
from exoshow.axis import Axes
from exoshow.layer import Layer


class ExoShow:
    def __init__(self,
                 db_dir=None,
                 out_dir=None,
                 db_date=None,
                 include_ss=True,
                 xdata='any_a',
                 ydata='any_mass',
                 title=None,
                 marker='x',
                 marker_size=1.2,
                 color='k',
                 legend_color='k',
                 legend_title=None,
                 ):

        if db_dir is None:
            db_dir = Path(__file__).resolve().parent.parent / 'db'

        self.date, self.db_exoplanet = db.read_exoplanets(db_dir, date=db_date)

        self.db_subset = self.db_exoplanet.copy()
        self.db_ss = db.read_solar_system()

        if out_dir is None:
            out_dir = Path(__file__).resolve().parent.parent / 'output' / self.date
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir

        self.axes = Axes(xdata, ydata, self.db_subset,
                         color=color,
                         title=title,
                         )

        self.layers: list[Layer] = []

        if legend_title is None:
            legend_title = f"As of {self.date[:4]}.{self.date[4:6]}.{self.date[6:8]}"

        self.add_layer(marker=marker,
                       color=color,
                       marker_size=marker_size,
                       legend_color=legend_color,
                       legend_title=legend_title,
                       )

        if include_ss:
            self.ss_layer = self.add_layer_ss()
        else:
            self.ss_layer = None

    def fix_xlims(self, value=None, margin=None):
        """
Fixes xlim according to specific min-max or to the current range of dataset
        Parameters
        ----------
        value: None, (float, float)
           Either None to take limits from current database, or explicit (min,max)
        margin: None, float
           The margin to use for plotting. if None then use matplotlib.rcparam['margin']

        Returns
        -------
        object
        """
        if value is None:
            value = self.db_subset
        self.axes.fix_xlims(value, margin=margin)

    def fix_ylims(self, value=None, margin=None):
        """
Fixes ylim according to specific min-max or to the current range of dataset
        Parameters
        ----------
        value: None, (float, float)
           Either None to take limits from current database, or explicit (min,max)
        margin: None, float
           The margin to use for plotting. if None then use matplotlib.rcparam['margin']

        Returns
        -------
        object
        """
        if value is None:
            value = self.db_subset
        self.axes.fix_ylims(value, margin=margin)

    def fix_lims(self, margin=None):
        self.fix_xlims(margin=margin)
        self.fix_ylims(margin=margin)

        return self

    def set_xaxis(self, label,
                  string=None,
                  logarithmic=None,
                  inverted=False,
                  ):
        self.axes.set_xdata(label, string=string, logarithmic=logarithmic, inverted=inverted)

    def set_yaxis(self, label,
                  string=None,
                  logarithmic=None,
                  inverted=False,
                  ):
        self.axes.set_ydata(label, string=string, logarithmic=logarithmic, inverted=inverted)

    def add_layer_ss(self, zorder=20, **kwargs):
        return self.add_layer(zorder=zorder, store=False, images="index", **kwargs)

    def add_layer(self, zorder=1, store=True,
                  filter_by_name=None,
                  **kwargs,
                  ) -> Layer:
        layer = Layer(ids=filter_by_name,
                      zorder=zorder,
                      **kwargs,
                      )

        if store:
            self.layers.append(layer)

        return layer

    def del_layer(self, position=-1):
        """
        Delete layer

        Parameters
        ----------
        position: int
        id of layer to delete. if value is larger than the number of layers, then it will delete the last one.

        Returns
        -------

        """
        if -len(self.layers) <= position < len(self.layers):
            position = -1
        self.layers.pop(position)

    ######################
    #
    # Closing up
    #
    ######################

    def show(self):
        self._plot().show()

    def save(self, filename, postfix=".png",
             out_dir=None,
             ):
        f = self._plot()

        filename = self.out_dir/filename

        if not len(Path(filename).suffixes):
            filename = f"{str(filename)}{postfix}"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        f.savefig(filename)

    def _plot(self) -> plt.Figure:
        """
Plots the figure. Before this call, no matplotlib command was issued.

        Returns
        -------
        the matplotlib.figure that holds the plot
        """
        ax = self.axes.plot()

        for layer in self.layers:
            layer.plot_axes_df(self.axes, self.db_subset)

        if self.ss_layer is not None:
            self.ss_layer.plot_axes_df(self.axes, self.db_ss)

        return ax.get_figure()

    #######################################
    #
    # DB functions
    #
    #######################################

    def db_reset(self):
        self.db_subset = self.db_exoplanet.copy()
        return self

    def before(self, year, reset=False, legend_title=None):
        if reset:
            self.db_reset()
        if legend_title is not None:
            self.layers[0].legend_title = legend_title
        self.db_subset = self.db_subset.loc[self.db_subset['discovered'].astype(float) <= year]
        return self

    def filter_str(self, **kwargs):
        for key, value in kwargs.items():
            self.db_subset = self.db_subset.loc[self.db_subset[key].str.contains(value, case=False)]

        return self

    def db_add_column(self,
                      permanent=False,
                      **kwargs,
                      ):

        df = db.add_column(self.db_exoplanet if permanent else self.db_subset, **kwargs)

        if permanent:
            self.db_exoplanet = df.copy()
        self.db_subset = df

    def db_compute(self,
                   permanent=False,
                   **kwargs,
                   ):

        df = db.compute(self.db_exoplanet if permanent else self.db_subset, **kwargs)

        if permanent:
            self.db_exoplanet = df.copy()
        self.db_subset = df
