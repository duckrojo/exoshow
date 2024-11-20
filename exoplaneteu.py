# import procastro as pa
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import astropy.constants as c

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype
import re
from glob import glob
import os


def matches_column(data, column, value):
    return data.loc[data[column].str.contains(value, case=False)]


def detected_before(data, year):
    return data.loc[data["discovered"].astype(float) <= year]


def set_log_ticks(ax, axis):
    orig_lims = getattr(ax, f"get_{axis}lim")()
    lims = np.array(orig_lims)
    lims = np.array(lims).astype(int)+np.array([0, 1])
    dtick = ((lims[1]-lims[0])/5 + 0.5).astype(int)
    if dtick == 0:
        dtick = 1
    tick_vals = lims[0]+dtick*np.arange(int((lims[1]-lims[0])/dtick)+1)
    ticks = tick_vals
    labels = [f"$10^{{{t}}}$" for t in tick_vals]
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")(labels)
    minor = np.log10(np.array([np.array([1, 2, 3, 4, 5, 6, 7, 8, 9.0])*(10.0**major)
                               for major in range(ticks[0]-1, ticks[-1])]).flatten())
    getattr(ax, f"set_{axis}ticks")(minor, minor=True)
    getattr(ax, f"set_{axis}lim")(orig_lims)


def replot_data(function):
    def wrapper(*args, **kwargs):
        redraw = kwargs.pop('redraw', True)
        f = function(*args, **kwargs)
        if redraw:
            args[0].reset_figure()
            print(f"reploting data here")
        return f
    return wrapper


class MassAxis:

    def __init__(self,
                 frame_color='black',
                 color="black", marker=None, marker_size=40,
                 before=None, method=None, targets=None,
                 highlight=None, annotate=None,
                 xlim=None, ylim=None,
                 ss_marker='s', ss_marker_size=50, ss_marker_color='r',
                 nn_marker='x', nn_marker_size=50, nn_marker_color='magenta',
                 hl_marker=None, hl_marker_size=None, hl_marker_color=None,
                 legend_location="lower right",
                 show_legend=True, show_molecules=False,
                 x_axis="any_a", y_axis="any_mass", z_axis=None,
                 x_scale='log', y_scale='log', z_scale='log',
                 compute=None, extra_info=None,
                 angles_3d=None, savefig=None,
                 ):

        # properties initialization
        default_color_dict = {'detection_type': {"transit": 'blue', "imaging": 'red',
                                                 "radial velocity": 'green', "microlensing": 'yellow',
                                                 "astrometry": 'orange', "timing": 'cyan',
                                                 "TTV": 'purple',
                                                 }
                              }
        default_marker_dict = {'detection_type': {"transit": "x", "imaging": "^", "radial velocity": ".",
                                                  "microlensing": "*", "astrometry": "s",
                                                  "timing": "v", "TTV": "o",
                                                  }
                               }
        self.props = {'frame_color': frame_color,
                      'marker_size': marker_size,
                      'ss_marker': ss_marker,
                      'ss_marker_size': ss_marker_size,
                      'ss_marker_color': ss_marker_color,
                      'nn_marker': nn_marker,
                      'nn_marker_size': nn_marker_size,
                      'nn_marker_color': nn_marker_color,
                      'hl_marker': hl_marker,
                      'hl_marker_size': hl_marker_size,
                      'hl_marker_color': hl_marker_color,
                      'show_legend': show_legend,
                      'show_molecules': show_molecules,
                      'legend_location': legend_location,
                      'annotate': annotate,
                      'save_prefix': 'fig',
                      'db_date': None,
                      'defaults_discrete': {'color': default_color_dict,
                                            'marker': default_marker_dict},
                      }

        self.axfig = [plt.figure(figsize=(10, 8)), None]

        self.plot3d = False
        self._info_axis = {'x': None, 'y': None, 'z': None}
        self._axis_scales = {'x': None, 'y': None, 'z': None}
        self._info_marker = None
        self._info_color = None

        self._keep = {'color': None,
                      'marker': None,
                      'marker_size': None,
                      'annotate': None,
                      'xlim': None,
                      'ylim': None,
                      }

        # read exoplanet and Solar System db
        self.db_exoplanet = self.read_exoplanets()
        self.db_subset = self.db_exoplanet.copy()

        self.db_ss = None
        self.read_solar_system()
        if extra_info is not None:
            self.add_column(**extra_info)
        if compute is not None:
            self.compute(**compute)

        # default legend title
        self.legend_title = f"As of 20{self.props['db_date'][0:2]}/" +\
                            f"{self.props['db_date'][2:4]}/{self.props['db_date'][4:6]}"
        self._legend = None

        # apply filters
        if before is not None:
            self.set_before(before, redraw=False)
        if method is not None:
            self.set_method(method, redraw=False)
        if targets is not None:
            self.set_targets(targets, redraw=False)

        # those to highlight
        self.highlights = []
        self.add_highlight(highlight)

        # initialize axes, color and marker, which resets figures
        self.set_data_marker(marker, redraw=False)
        self.set_data_color(color, redraw=False)
        self.set_axes([x_axis, y_axis, z_axis],
                      [x_scale, y_scale, z_scale],
                      )

        self.view3d(angles_3d)
        self.set_lims(xlim, ylim)

        if savefig:
            self.props['save_prefix'] = savefig
            self.save()

    ################
    #
    # Plotting  calls
    #
    @replot_data
    def set_axes(self, axes_names=None, axes_scale=None,
                 x_axes=None, y_axes=None, z_axes=None,
                 x_scale=None, y_scale=None, z_scale=None,
                 force2d=False):

        def init(user_name, user2_name, self_name):
            names = ['x', 'y', 'z']
            if user_name is None:
                user_name = user2_name
            setattr(self, self_name, {name: getattr(self, self_name)[name] if column is None else column
                                      for name, column
                                      in zip(names, user_name)})

        init(axes_names, [x_axes, y_axes, z_axes], '_info_axis')
        init(axes_scale, [x_scale, y_scale, z_scale], '_axis_scales')

        if self._info_axis['z'] is not None:
            self.plot3d = True
        if force2d:
            self.plot3d = False

    def reset_figure(self):
        fig = self.axfig[0]
        fig.clf()

        # figure initialization
        projection = '3d' if self.plot3d else None
        self.axfig[1] = fig.add_subplot(111, projection=projection)

        self._fill_data()
        self._setup_aesthetics()

    def view3d(self, angles_3d):
        if self.plot3d and angles_3d is not None and len(angles_3d) == 2:
            self.axfig[1].view_init(*angles_3d)

    def legend(self, title_fontsize=14, loc=None):
        try:
            self._legend = self.axfig[1].legend(title=self.legend_title,
                                                title_fontsize=title_fontsize, loc=loc)
        except UserWarning:
            pass

    def set_lims(self,
                 xlim=None,
                 ylim=None,
                 axis=None,
                 lims=None,
                 verbose=True,
                 ax=None,
                 ):
        """
        Set limits of the plot

        Parameters
        ----------
        xlim:
           X limits
        ylim:
           Y limit TwoTuple
        axis:
           if set to 'x', 'y', or 'z', then look for TwoTuple of limits in lims
        lims:
           TwoTuple of limit if `axis` keyword is set
        verbose:
           whether to print output
        ax
        """
        if ax is None:
            ax = self.axfig[1]

        axis_list = {}
        if axis is not None:
            axis_list[axis] = lims
        if xlim is not None:
            axis_list['x'] = xlim
        if ylim is not None:
            axis_list['y'] = ylim

        for axis, lims in axis_list.items():
            getattr(ax, f'set_{axis}lim')(np.log10(lims) if self._axis_scales[axis] == 'log' else lims)

        if verbose:
            for axis, col in self._info_axis.items():
                if col is not None:
                    lims = np.array(getattr(ax, f'get_{axis}lim')())
                    print(f"{axis}lim ({col:10s}) = {10.0**lims if self._axis_scales[axis] == 'log' else lims}")

    @replot_data
    def set_data_marker(self, marker, use_marker_dict=None):
        self._info_marker = marker
        self.add_default_marker(marker, use_marker_dict)

    @replot_data
    def set_data_color(self, color, use_color_dict=None):
        self._info_color = color
        self.add_default_color(color, use_color_dict)

    def add_default_marker(self, marker, marker_dict):
        self.props['defaults_discrete']['marker'][marker] = marker_dict

    def add_default_color(self, color, color_dict):
        self.props['defaults_discrete']['color'][color] = color_dict

    ######
    #
    # Filters
    #
    @replot_data
    def set_before(self, before, reset=False):
        self.db_subset = detected_before(self.db_exoplanet if reset else self.db_subset,
                                         before)
        self.legend_title = f"Up to {before}"

    @replot_data
    def set_method(self, method, reset=False):
        self.db_subset = matches_column(self.db_exoplanet if reset else self.db_subset,
                                        "detection_type", method)

    @replot_data
    def set_targets(self, targets):
        self.db_subset = self.db_subset.loc[targets]

    @replot_data
    def reset_subset(self):
        self.legend_title = f"As of 20{self.props['db_date'][0:2]}/" +\
                            f"{self.props['db_date'][2:4]}/{self.props['db_date'][4:6]}"
        self.db_subset = self.db_exoplanet

    def add_highlight(self, highlight, reset=False):
        if reset:
            self.highlights = []
        if not isinstance(highlight, (list, tuple)):
            highlight = [highlight]

        self.highlights.extend(highlight)

    #####################
    #
    # Plotting  internals
    #
    def _array_to_plot(self,
                       color, marker, marker_size,
                       data=None, label='',
                       annotate=None, annotate_size=7,
                       images=None, image_size=0.09,
                       zorder=None):
        color = self._get_color(color)

        if data is None:
            data = self.db_subset
        dataaxis = []
        for axis in ['x', 'y', 'z']:
            col = self._info_axis[axis]
            if col is not None:
                dataaxis.append(np.log10(data[col]) if self._axis_scales[axis] == 'log' else data[col])

        if isinstance(color, (list, tuple)) and len(color) == 2:
            if color[1] is None:
                color = data[color[0]]

        self.axfig[1].scatter(*dataaxis,
                              marker=marker, label=label, s=marker_size, c=color,
                              zorder=zorder)

        if annotate is not None and not self.plot3d:
            if annotate == 'index':
                names = data.index
            else:
                names = data[annotate]

            for x, y, lab in zip(dataaxis[0], dataaxis[1], names):
                if not np.isnan(x) and not np.isnan(y):
                    self.axfig[1].annotate(lab, (x, y), size=annotate_size)

        if images is not None and not self.plot3d:
            if images == 'index':
                filenames = data.index
            else:
                filenames = data[images]

            for x, y, lab in zip(dataaxis[0], dataaxis[1], filenames):
                img = np.asarray(Image.open(f'images/{lab.lower()}.png'))
                axx = self.axfig[1]
                x_axis, y_axis = self.axfig[0].transFigure.inverted().transform(axx.transData.transform((x, y)))

                pl_ax = self.axfig[0].add_axes((x_axis - image_size/2, y_axis - image_size/2,
                                                image_size, image_size),
                                               zorder=zorder)
                pl_ax.axis('off')
                pl_ax.imshow(img)

    def plot_highlight(self, color='green', marker='o', marker_size=50, use_subset=False):
        if marker is None:
            marker = self.props['hl_marker']
        if marker_size is None:
            marker_size = self.props['hl_marker_size']
        if color is None:
            color = self.props['hl_marker_color']

        if use_subset:
            db = self.db_subset
        else:
            db = self.db_exoplanet
        plot_from_ss = []
        plot_from_exop = []
        for cm in self.highlights:
            if cm in self.db_ss.index:
                plot_from_ss.append(cm)
            elif cm in db.index:
                plot_from_exop.append(cm)
            elif cm is not None:
                print(f"Cannot find planet '{cm}' in {'filtered ' if use_subset else ''}database to highlight")

        if len(plot_from_ss):
            self._array_to_plot(color, marker, marker_size,
                                data=self.db_ss.loc[plot_from_ss], label='')
        if len(plot_from_exop):
            self._array_to_plot(color, marker, marker_size,
                                data=self.db_exoplanet.loc[plot_from_exop], label='',
                                )

    def plot_notna(self, column, color=None, symbol=None, marker_size=None):
        if symbol is None:
            symbol = self.props['nn_marker']
        if color is None:
            color = self.props['nn_marker_color']
        if marker_size is None:
            marker_size = self.props['nn_marker_size']

        self.db_subset = self.db_subset.loc[self.db_subset[column].notna()]
        print(f"with {column} detected: {len(self.db_subset)}")
        self.plot_data(color=color, marker=symbol, marker_size=marker_size)

    def plot_ss(self, marker=None, marker_size=None, color=None):
        if marker is None:
            marker = self.props['ss_marker']
        if marker_size is None:
            marker_size = self.props['ss_marker_size']
        if color is None:
            color = self.props['ss_marker_color']

        try:
            self._array_to_plot(color, marker, marker_size,
                                data=self.db_ss, zorder=15,
                                images='index')

            # array_to_plot(self, data=None, label='', color=None, mark=None, marker_size=None):
            # self.axfig[1].scatter(*axisdata, marker=ss_marker, s=ss_marker_size, c=ss_marker_color)
            # plot_crossmarks(ax, ss, ['JUPITER', 'EARTH'], [x_axis, y_axis, z_axis],
            #                 [x_scale, y_scale, z_scale])
        except KeyError:
            third_axis = f', {self._info_axis["z"]}' if self._info_axis["z"] is not None else ''
            all_axis = f"{self._info_axis['x']}, {self._info_axis['y']}{third_axis}"
            print(f"warning: One of the axis ({all_axis}) was not found on Solar System data, skipping")

    def plot_data(self,
                  color=None,
                  marker=None,
                  marker_size=None,
                  annotate=None,
                  xlim=None,
                  ylim=None,
                  keep=False,
                  ):
        if keep:
            xlim = self._keep['xlim']
            ylim = self._keep['ylim']
            color = self._keep['color']
            marker = self._keep['marker']
            marker_size = self._keep['marker_size']
            annotate = self._keep['annotate']

        self._keep = {'color': color,
                      'marker': marker,
                      'marker_size': marker_size,
                      'annotate': annotate,
                      'xlim': xlim,
                      'ylim': ylim,
                      }

        if color is None:
            color = self._info_color
        if marker is None:
            marker = self._info_marker
        marker = self._get_marker(marker)
        if marker_size is None:
            marker_size = self.props['marker_size']
        if annotate is None:
            annotate = self.props['annotate']

        indata = self.db_subset

        self.set_lims(xlim=xlim, ylim=ylim)

        if isinstance(marker, str):
            self._array_to_plot(color, marker, marker_size,
                                data=indata, label="", annotate=annotate)
        else:
            marker_column = marker[0]
            markers = marker[1]
            for value, mark in markers.items():
                data = matches_column(indata, marker_column, value)

                self._array_to_plot(color, mark, marker_size,
                                    data=data, label=f"{value} ({len(data)})",
                                    annotate=annotate)

    def _fill_data(self):
        # scatter plot
        self.plot_data(keep=True)

        # present legend
        print(f"Plotted {self.legend_title}, in total {len(self.db_subset)} planets ")
        if self.props['show_legend']:
            self.legend(title_fontsize=14, loc=self.props['legend_location'])

        # add with no legend: molec and ss
        if self.props['show_molecules']:
            self.plot_notna("molecules", color="red",
               symbol='o', marker_size=80)
        if self.db_ss is not None:
            self.plot_ss()

        # highlight few
        self.plot_highlight()

    def _setup_aesthetics(self):
        # aesthetics
        color = self.props['frame_color']
        self._format_axis(color)
        if color is not None:
            self._frame_color(color)

    def _frame_color(self, color, ax=None):
        if ax is None:
            ax = self.axfig[1]
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.yaxis.label.set_color(color)
        ax.xaxis.label.set_color(color)
        plt.setp(plt.getp(ax, 'xticklabels'), color=color)
        plt.setp(plt.getp(ax, 'yticklabels'), color=color)
        ax.title.set_color(color)
        if self._legend is not None:
            self._legend.get_title().set_color(color)
            for t in self._legend.get_texts():
                t.set_color(color)

    def _format_axis(self, color):
        title_forward = {'any_mass': [r"Mass ($M_J$)", r"Mass ($M_\oplus$)",
                                      lambda x: np.array(x)*c.M_jup/c.M_earth,
                                      'log'],
                         'any_a': ["Semi-major axis (AU)", r"Period around 1 $M_{\odot}$ star (days)",
                                   lambda x: (np.array(x)**1.5)*365.24,
                                   'log'],
                         'planet_temp': ["Temperature (K)", None],
                         'planet_flux': [r"Flux received ($F_\oplus$)", "Equilibrium Temperature (K)",
                                         lambda x: (x*(c.L_sun/c.au/c.au/c.sigma_sb).value/16/np.pi)**0.25,
                                         'log'],
                         'transit_duration': ['Approximate transit duration (h)', None],
                         'planet_gravity': [r'Planet Gravity ($m/s^{2}$', "Compared to Earth",
                                            lambda x: x / 9.7, 'log'],
                         'transit_modulation': [r'$\lambda-to-\lambda$ transit modulation', None],
                         'transit_snr': ['Expected SNR (arbitrary scale)', None],
                         }

        self._frame_color(color)

        axis_scale = self._axis_scales
        for axis, axis_input in self._info_axis.items():
            def set_title_scale(ax, scale, label):
                if label is not None:
                    getattr(ax, f'set_{axis}label')(label)
                if scale == 'log':
                    set_log_ticks(ax, axis)

            if axis_input is not None:
                try:
                    titles = title_forward[axis_input]
                except KeyError:
                    titles = [axis_input, None]
                set_title_scale(self.axfig[1], axis_scale[axis], titles[0])

                if titles[1] is not None and not self.plot3d:
                    # if configured opposite axis and 2D
                    forward = titles[2] if axis_scale[axis] != 'log' else lambda x: titles[2](10**x)

                    ax2 = getattr(self.axfig[1], f"twin{'y' if axis == 'x' else 'x'}")()
                    lims = forward(np.array(getattr(self.axfig[1], f'get_{axis}lim')()))
                    self.set_lims(axis=axis, lims=lims, ax=ax2, verbose=False)

                    self._frame_color(color, ax=ax2)
                    set_title_scale(ax2, titles[3], titles[1])

        return self

    def _get_color(self, color):
        if color is None:
            color = 'black'
        default_colors = self.props['defaults_discrete']['color']

        if isinstance(color, str):
            if color in self.db_subset.columns:
                if is_numeric_dtype(self.db_subset[color]):
                    color = [color, None]
                else:
                    raise NotImplemented("Need to implement discrete color assignment")

        return color

    def _get_marker(self, marker):

        default_markers = self.props['defaults_discrete']['marker']
        if marker is None:
            return "x"
        elif marker in default_markers.keys():
            return [marker, default_markers[marker]]
        elif marker in self.db_exoplanet.columns:
            raise NotImplemented("Need to create a default list of markers when not specified")

        return marker

    ##########
    #
    # Details
    #
    def __del__(self):
        self.close()

    def ion(self):
        plt.ion()
        self.show()

    def ioff(self):
        plt.ioff()

    def show(self):
        self.axfig[0].show()

    def close(self):
        plt.close(self.axfig[0])

    def save(self, filename=None, fig_directory=None):
        if filename is None:
            if fig_directory is None:
                fig_directory = self.props["db_date"]

            postfix = self.props['frame_color']
            filename = f"{fig_directory}/{self.props['save_prefix']}_{postfix}.png"
        self.axfig[0].savefig(filename)

    ###################
    #
    # Reading of databases
    #
    def read_exoplanets(self, directory='./', planet_albedo=0.1, t_sun=5777):
        filename = sorted(glob(directory+'exoplanet*csv'))[-1]

        rec = re.compile(r"_(\d+)[_.]")
        match = re.search(rec, filename)
        date_in_file = match.group(1)
        try:
            os.mkdir(f"{date_in_file}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"{date_in_file}/before")
        except FileExistsError:
            pass

        exop = pd.read_csv(filename, index_col='name')

        # Using Keppler a(P,M)
        exop['semi_major_axis'].fillna(((exop['orbital_period']/365.24)**2 * exop['star_mass'])**(-1/3))
        # Using equilibrium flux F(T_star,R_star,a) in comparison with Earth and assuming same albedo
        exop['planet_flux'] = (exop['star_teff'] / t_sun) ** 4 * (exop['star_radius'] / exop['semi_major_axis']) ** 2
        # Using equilibrium temperature F(T_star,R_star,a)
        exop['planet_temperature'] = np.sqrt(exop['star_radius']*c.R_sun/c.au/2.0/exop['semi_major_axis'] *
                                             np.sqrt(1-planet_albedo))*exop['star_teff']

        exop.loc[:, 'any_mass'] = exop['mass'].copy()
        exop.loc[:, 'any_a'] = exop['semi_major_axis'].copy()
        exop.fillna({'any_a': (exop['star_mass']*exop['orbital_period']**2)**(1/3),
                     'any_mass': exop['mass_sini'],
                     },
                    inplace=True)

        exop['inversions'] = False
        for name in ['WASP-33 Ab', 'WASP-18 Ab', 'WASP-121 b']:
            if name not in exop.index:
                raise ValueError(f"Planet with inversion f{name} not found in this exoplanet "
                                 f"version {self.props['db_date']}")
            exop.loc[name, 'inversions'] = True

        exop['transit_duration'] = (2*exop['star_radius']*c.R_sun /
                                    (2*np.pi*exop['any_a']*c.au/(exop['orbital_period']*24)))

        exop['planet_gravity'] = c.G * exop['any_mass'] * c.M_jup / (exop['radius'] * c.R_jup)**2

        exop['planet_density'] = 3*exop['any_mass']/(4*np.pi*exop['radius']**3)

        exop['mean_molecular_mass'] = 2.22 + (29-2.22)*(exop['planet_density'] > 3)

        exop['scale_height'] = c.k_B * exop['planet_temperature'] / \
                               (exop['mean_molecular_mass'] * c.m_p * exop['planet_gravity'])

        exop['transit_modulation'] = exop['radius'] * exop['scale_height'] / exop['star_radius']**2

        exop['transit_depth'] = (exop['radius']*c.R_jup/exop['star_radius']/c.R_sun)**2

        exop['transit_snr'] = exop['transit_modulation'] * 10**(-0.2*exop['mag_v']) * np.sqrt(exop['transit_duration'])

        self.props["db_date"] = date_in_file
        return exop

    def read_solar_system(self, filename='solar_system.csv', planet_albedo=0.3, t_sun=5777):
        ss = pd.read_csv(filename, index_col='Name').drop(index="PLUTO")
        # Solar System mass in Jup's units
        ss["any_mass"] = ss["Mass"] / ss["Mass"].loc["JUPITER"]
        ss['any_a'] = ss['Distance from Sun'].copy()
        ss['planet_temp'] = np.sqrt(c.R_sun / c.au / 2.0 / ss['any_a'] * np.sqrt(1 - planet_albedo)) * t_sun
        ss['planet_flux'] = 1 / ss['any_a'] ** 2

        self.db_ss = ss

    def add_column(self, **kwargs):
        for column, content in kwargs.items():
            self.db_exoplanet[column] = np.nan
            for name, value in content.items():
                self.db_exoplanet.at[name, column] = value

    def compute(self, **kwargs):
        for column, formula in kwargs.items():
            formula = formula.replace('<index>', 'self.db_exoplanet.index')
            formula = formula.replace('<', 'self.db_exoplanet["')
            formula = formula.replace('>', '"]')
            self.db_exoplanet[column] = eval(formula)
