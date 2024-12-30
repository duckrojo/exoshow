from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from exoshow.definitions import default_color_dict, default_marker_dict
from exoshow.axis import Axes


def _get_discrete_dicts(df, label, dictionary, columns, default):
    if label in columns:
        output = [lab.lower() for lab in df[label]]
        if label not in dictionary.keys():  # If no default marker exist for that column
            out_dict = {k: v for k, v in zip(set(output), default)}
        else:
            out_dict = {k.lower(): v for k, v in dictionary[label].items()}
    else:
        output = np.array([None] * len(df))
        out_dict = {None: label}
    return output, out_dict


class Layer:
    def __init__(self,
                 mask: list[str] | None = None,
                 marker: str = 'x',
                 color: str = 'k',
                 marker_size: float = 1,
                 size_extreme: tuple[float, float] | None = None,
                 zorder=None,
                 images: str | None = None,
                 image_size: float = 9,
                 annotate: str | None = None,
                 annotate_size: int = 9,
                 legend_color: str = 'k',
                 legend_position: str = 'lower right',
                 legend_title: str = None,
                 legend_fontsize: int = 9,
                 marker_dicts: dict | None = None,
                 color_dicts: dict | None = None,
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
        mask: [str, ...], None
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
        self.ids = mask

        self.zorder = zorder

        self.images = images
        self.image_size = image_size

        self.legend_color = legend_color
        self.legend_position = legend_position
        self.legend_title = legend_title
        self.legend_fontsize = legend_fontsize

        if size_extreme is None:
            self.size_extreme = [1, 5]
        else:
            self.size_extreme = size_extreme

        self.annotate_size = annotate_size
        self.annotate = annotate
        self.marker_size = marker_size
        self.marker = marker
        self.color = color
        self.marker_dicts = marker_dicts or default_marker_dict
        self.color_dicts = color_dicts or default_color_dict

    def props(self):
        elements = ['zorder', 'images', 'image_size',
                    'legend_color', 'legend_position', 'legend_title', 'legend_fontsize',
                    'size_extreme', 'annotate_size', 'annotate',
                    'marker_size', 'marker', 'color',
                    'marker_dicts', 'color_dicts']

        return {k: getattr(self, k) for k in elements}

    def __hash__(self):
        return (self.ids,
                self.images, self.image_size,
                self.marker, self.size_extreme, self.marker_size,
                self.annotate_size, self.annotate,
                self.color)

    def plot_axes_df(self,
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

        valid = df if len(self.ids) != len(df) or self.ids is None else df.loc[self.ids]
        columns = valid.columns.tolist()

        nn = len(valid)
        xx = horizontal.scale(valid[xlabel])
        yy = vertical.scale(valid[ylabel])

        if self.images is not None and not no_images:
            if self.images == 'index':
                filenames = df.index
            else:
                filenames = df[self.images]

            image_size = self.image_size

            for x, y, lab in zip(xx, yy, filenames):
                img = np.asarray(Image.open(Path(__file__).parent.parent/f'images/{lab.lower()}.png'))

                f = ax.figure

                x_axis, y_axis = f.transFigure.inverted().transform(ax.transData.transform((x, y)))

                pl_ax = f.add_axes((x_axis - image_size/200, y_axis - image_size/200,
                                    image_size/100, image_size/100),
                                   zorder=self.zorder)
                pl_ax.axis('off')
                pl_ax.imshow(img)

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
            marker_size_scaled = np.array([3*((x1 - x0) * self.marker_size / 200)**2] * nn)

        for m in marker_dict.keys():
            if isinstance(m, str):
                m = m.lower()
            group = np.array([m is mk or m in mk for mk in marker])
            if group.sum() == 0:
                continue
            label = f"{m} ({group.sum()})"
            ax.scatter(xx[group], yy[group],
                       marker=marker_dict[m],
                       c=[color_dict[c] for c in color[group]],
                       s=list(marker_size_scaled[group]),
                       zorder=self.zorder,
                       label=label,
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
            legend = ax.legend(loc=self.legend_position,
                               title=self.legend_title,
                               title_fontsize=self.legend_fontsize,
                               fontsize=self.legend_fontsize,
                               )

            legend.get_title().set_color(self.legend_color)
            for t in legend.get_texts():
                t.set_color(self.legend_color)
