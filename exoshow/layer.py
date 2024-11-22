import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from exoshow import default_marker_dict, default_color_dict
from exoshow.axis import Axes


def _get_discrete_dicts(df, label, dictionary, columns, default):
    if label in columns:
        output = df[label]
        if label not in dictionary.keys():  # If no default marker exist for that column
            out_dict = {k: v for k, v in zip(set(output), default)}
        else:
            out_dict = dictionary[label]
    else:
        output = [None] * len(df)
        out_dict = {None: label}
    return output, out_dict


class Layer:
    def __init__(self,
                 figure: plt.Figure,
                 ids: list[str] | None = None,
                 marker: str = 'x',
                 color: str = 'k',
                 marker_size: float = 1,
                 size_extreme: tuple[float, float] | None = None,
                 zorder=None,
                 images: str | None = None,
                 image_size: float = 0.9,
                 annotate: str | None = None,
                 annotate_size: int = 9,
                 ):
        """

        Parameters
        ----------
        zorder:
            the higher, the more on top the image appears
        images:
            to use images, set the str to column value
        image_size:
            percent extent of images
        figure: plt.Figure
            matplotlib figure where the layer will be drawn
        ids: [str, ...], None
            name of targets that can be drawn in this layer. If None, all targets will be drawn/
        marker: str
            name of column to group for markers or marker to use
        color: str
            column with color info, or color value
        marker_size: str|float
           name of column with size info (to be scaled accoridng to size_extreme), or size of marker in percent of axis
        size_extreme: (float, float)
           Min and Max size of the marker in percent if taken from column value
        """
        self.ids = ids

        self.zorder = zorder

        self.images = images
        self.image_size = image_size
        self.image_axes = []

        if size_extreme is None:
            self.size_extreme = [1, 5]
        else:
            self.size_extreme = size_extreme

        self.annotate_size = annotate_size
        self.annotate = annotate
        self.marker_size = marker_size
        self.marker = marker
        self.color = color
        self.marker_dicts = default_marker_dict
        self.color_dicts = default_color_dict

        self.ax = figure.add_subplot(111)
        self.ax.axis('off')

    def plot_images(self, df, horizontal, vertical):
        if self.images == 'index':
            filenames = df.index
        else:
            filenames = df[self.images]

        xlabel = horizontal.label
        ylabel = vertical.label

        self.ax.set_xlims(horizontal.limits)
        self.ax.set_ylims(vertical.limits)

        valid = df if self.ids is None else df.loc[df.index.isin(self.ids)]
        image_size = self.image_size

        if len(self.image_axes):
            for ax in self.image_axes:
                ax.remove()

        for x, y, lab in zip(valid[xlabel], valid[ylabel], filenames):
            img = np.asarray(Image.open(f'images/{lab.lower()}.png'))

            axx = self.ax
            x_axis, y_axis = axx.transFigure.inverted().transform(axx.transData.transform((x, y)))

            pl_ax = self.ax.add_axes((x_axis - image_size/2, y_axis - image_size/2,
                                      image_size, image_size),
                                     zorder=self.zorder)
            pl_ax.axis('off')
            pl_ax.imshow(img)
            self.image_axes.append(pl_ax)

    def plot(self,
             df: pd.DataFrame,
             horizontal: Axes,
             vertical: Axes,
             no_images: bool = False,
             ):
        """
        Create the layer using updated values of horizontal and vertical axes as well as latest dataframe

        Parameters
        ----------
        no_images: bool
           if True, then it will not show images even if set
        df
        horizontal
        vertical

        Returns
        -------

        """

        if self.images is not None and not no_images:
            return self.plot_images(df, horizontal, vertical)

        xlabel = horizontal.label
        ylabel = vertical.label

        xlims = horizontal.limits
        ylims = vertical.limits
        self.ax.set_xlims(xlims)
        self.ax.set_ylims(ylims)

        valid = df if self.ids is None else df.loc[df.index.isin(self.ids)]
        columns = valid.columns.tolist()

        nn = len(valid)
        xx = valid[xlabel]
        yy = valid[ylabel]

        def data_to_points(points):
            return self.ax.dpi_scale_transform.inverted().transform(self.ax.transData.transform(points))*72

        marker, marker_dict = _get_discrete_dicts(valid, self.marker, self.marker_dicts, columns,
                                                  ["x", "^", ".", "*", "s", "v", "o"])
        color, color_dict = _get_discrete_dicts(valid, self.color, self.color_dicts, columns,
                                                ['blue', 'red', 'green', 'yellow',
                                                 'orange', 'cyan', 'purple', ]
                                                )

        extremes = data_to_points([(xlims[0], ylims[0]), (xlims[1], ylims[1])])
        (x0, y0), (x1, y1) = extremes
        min_p = (x1 - x0) * self.size_extreme[0] / 100
        max_p = (x1 - x0) * self.size_extreme[1] / 100
        if self.marker_size in columns:
            marker_size = valid[self.marker_size]
            min_v = min(marker_size)
            max_v = max(marker_size)

            marker_size_scaled = min_p + (marker_size - min_v) * (max_p - min_p) / (max_v - min_v)

        else:
            marker_size_scaled = np.array([(x1 - x0) * self.marker_size / 100] * nn)

        for m in marker_dict.keys():
            group = marker == m
            self.ax.scatter(xx[group], yy[group],
                            marker=marker_dict[m],
                            color=color[group], s=marker_size_scaled[group],
                            zorder=self.zorder,
                            label=m,
                            )

        if self.annotate is not None:
            if self.annotate == 'index':
                names = df.index
            else:
                names = df[self.annotate]

            for x, y, lab in zip(xx, yy, names):
                if not np.isnan(x) and not np.isnan(y):
                    self.ax.annotate(lab, (x, y), size=self.annotate_size)

    def axes(self,
             horizontal: Axes,
             vertical: Axes,
             ):
        self.ax.axis('on')
        horizontal.in_axes(self.ax)
        vertical.in_axes(self.ax)
        self.ax.title.set_color(self.color)

        # todo: check legend
        if self._legend is not None:
            self._legend.get_title().set_color(color)
            for t in self._legend.get_texts():
                t.set_color(color)

