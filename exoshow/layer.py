from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.pyplot import axis

from exoshow.definitions import default_color_dict, default_marker_dict
from exoshow.axis import Axes


def _get_discrete_dicts(df, label, dictionary, columns, default):
    if label in columns:
        output = df[label]
        if label not in dictionary.keys():  # If no default marker exist for that column
            out_dict = {k: v for k, v in zip(set(output), default)}
        else:
            out_dict = dictionary[label]
    else:
        output = np.array([None] * len(df))
        out_dict = {None: label}
    return output, out_dict


class Layer:
    def __init__(self,
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
                 legend_color: str = 'k',
                 legend_position: str = 'lower right',
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
        ids: [str, ...], None
            name of targets that can be drawn in this layer. If None, all targets will be drawn/
        marker: str
            name of column to group for markers or marker to use
        color: str
            column with color info, or color value
        marker_size: str|float
           name of column with size info (to be scaled according to size_extreme), or size of marker in percent of axis
        size_extreme: (float, float)
           Min and Max size of the marker in percent if taken from column value. By default, 1%-5%
        """
        self.ids = ids

        self.zorder = zorder

        self.images = images
        self.image_size = image_size
        self.image_axes: list[axis] = []
        self.legend_color = legend_color
        self.legend_position = legend_position

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

    def __hash__(self):
        return (self.ids,
                self.images, self.image_size,
                self.marker, self.size_extreme, self.marker_size,
                self.annotate_size, self.annotate,
                self.color)

    def plot(self,
             axes: Axes,
             df: pd.DataFrame,
             no_images: bool = False,
             ):
        """
        Create the layer using updated values of horizontal and vertical axes as well as latest dataframe

        Parameters
        ----------
        axes:
           Axes information where to plot
        no_images: bool
           if True, then it will not show images even if set
        df

        Returns
        -------

        """

        ax = axes.new_ax()
        horizontal = axes.x_axes
        vertical = axes.y_axes

        xlabel = horizontal.label
        ylabel = vertical.label

        xlims = horizontal.limits
        ylims = vertical.limits
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        valid = df if self.ids is None else df.loc[df.index.isin(self.ids)]
        columns = valid.columns.tolist()

        nn = len(valid)
        xx = valid[xlabel]
        yy = valid[ylabel]

        if self.images is not None and not no_images:
            if self.images == 'index':
                filenames = df.index
            else:
                filenames = df[self.images]

            while len(self.image_axes):
                self.image_axes.pop().remove()

            image_size = self.image_size

            for x, y, lab in zip(valid[xlabel], valid[ylabel], filenames):
                img = np.asarray(Image.open(Path(__file__).parent.parent/f'images/{lab.lower()}.png'))

                f = ax.figure
                x_axis, y_axis = f.transFigure.inverted().transform(ax.transData.transform((x, y)))

                pl_ax = f.add_axes((x_axis - image_size/2, y_axis - image_size/2,
                                    image_size, image_size),
                                   zorder=self.zorder)
                pl_ax.axis('off')
                pl_ax.imshow(img)
                self.image_axes.append(pl_ax)
            return

        def data_to_points(points):
            return ax.figure.dpi_scale_trans.inverted().transform(ax.transData.transform(points))*72

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
            group = np.array([m.lower() in mk.lower() for mk in marker])
            ax.scatter(xx[group], yy[group],
                       marker=marker_dict[m],
                       c=list(color[group]), s=list(marker_size_scaled[group]),
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
                    ax.annotate(lab, (x, y), size=self.annotate_size)

        if self.legend_color and self.legend_position:
            legend = ax.legend(self.legend_position)

            legend.get_title().set_color(self.legend_color)
            for t in legend.get_texts():
                t.set_color(self.legend_color)
