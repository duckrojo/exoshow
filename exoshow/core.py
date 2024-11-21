# properties initialization
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as c

from exoshow import db

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

# for each column, it shows either a 4-tupe: title top/left, title bottom/right, formula
default_title = {'any_mass': ((r"Mass ($M_\oplus$)", 'log'),
                              (r"Mass ($M_J$)", lambda x: np.array(x) * c.M_jup / c.M_earth)
                              ),
                 'any_a': ((r"Period around 1 $M_{\odot}$ star (days)", 'log'),
                           ("Semi-major axis (AU)", lambda x: (np.array(x) ** 1.5) * 365.24)
                           ),
                 'planet_temp': (("Temperature (K)", None),
                                 ),
                 'planet_flux': (("Equilibrium Temperature (K)", 'log'),
                                 (r"Flux received ($F_\oplus$)",
                                  lambda x: (x * (c.L_sun / c.au / c.au / c.sigma_sb).value / 16 / np.pi) ** 0.25)
                                 ),
                 'transit_duration': (('Approximate transit duration (h)', None),
                                      ),
                 'planet_gravity': (("Compared to Earth", 'log'),
                                    (r'Planet Gravity ($m/s^{2}$', lambda x: x / 9.7)
                                    ),
                 'transit_modulation': ((r'$\lambda-to-\lambda$ transit modulation', 'linear'),
                                        ),
                 'transit_snr': (('Expected SNR (arbitrary scale)', 'linear'),
                                 )
                 }


class ExoShow:
    def __init__(self,
                 db_dir=None,
                 out_dir=None,
                 db_date=None,
                 include_ss=True,
                 ):

        if db_dir is None:
            db_dir = Path(__file__).resolve().parent.parent / 'db'

        date, self.db_exoplanet = db.read_exoplanets(db_dir, date=db_date)
        self.db_subset = self.db_exoplanet.copy()
        self.db_ss = db.read_solar_system()
        self.ss_props = dict(marker=None, marker_size=None, color=None, images=None)

        if out_dir is None:
            out_dir = Path(__file__).resolve().parent.parent / 'output' / date
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir

        self.include_ss = include_ss

        self._fig, self._ax = plt.subplot(111)


    def add_layer(self, zorder=1, props=None):


    def plot_ss(self, marker=None, marker_size=None, color=None, images=True):
        """
        Keeps previous use of parameters until reset

        Parameters
        ----------
        images
        marker
        marker_size
        color
        """
        if marker is None:
            marker = self.ss_props['marker']
        if marker_size is None:
            marker_size = self.ss_props['marker_size']
        if color is None:
            color = self.ss_props['color']
        if images is None:
            images = self.ss_props['images']
        self.ss_props = dict(marker=marker, marker_size=marker_size, color=color, images=images)

        self._array_to_plot(color, marker, marker_size,
                            data=self.db_ss, zorder=15,
                            images='index' if images else None)

        # try:
        #
        #     # array_to_plot(self, data=None, label='', color=None, mark=None, marker_size=None):
        #     # self.axfig[1].scatter(*axisdata, marker=ss_marker, s=ss_marker_size, c=ss_marker_color)
        #     # plot_crossmarks(ax, ss, ['JUPITER', 'EARTH'], [x_axis, y_axis, z_axis],
        #     #                 [x_scale, y_scale, z_scale])
        # except KeyError:
        #     third_axis = f', {self._info_axis["z"]}' if self._info_axis["z"] is not None else ''
        #     all_axis = f"{self._info_axis['x']}, {self._info_axis['y']}{third_axis}"
        #     print(f"warning: One of the axis ({all_axis}) was not found on Solar System data, skipping")

    ######################
    #
    # Closing up
    #
    ######################

    def save(self, filename, postfix=".png"):
        filename = self.out_dir/filename
        if not len(str(filename).suffixes):
            filename /= postfix
        filename.parent.mkdir(parents=True, exist_ok=True)

        self._ax.savefig(filename)

    #######################################
    #
    # DB functions
    #
    #######################################

    def db_reset(self):
        self.db_subset = self.db_exoplanet.copy()
        self.ss_props = {}

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

