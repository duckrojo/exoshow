# properties initialization
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as c

from exoshow import db
from exoshow.axis import Axes
from exoshow.layer import Layer

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
        self.layers: list[Layer] = []
        self.horizontal = Axes(self.db_subset, 'any_semi_major_axis', 'x')
        self.vertical = Axes(self.db_subset, 'any_mass', 'y')
        self.ss_layer = self.add_layer_ss()

        self._fig, self._ax = plt.subplot(111)

    def add_layer_ss(self, zorder=20, **kwargs):
        return self.add_layer(zorder=zorder, store=False, images="index", **kwargs)

    def add_layer(self, zorder=1, store=True,
                  filter_by_name=None,
                  marker='x', color='k',
                  marker_size: float = 1,
                  size_extreme: tuple[float, float] | None = None,
                  images: str | None = None,
                  image_size: float = 0.9,
                  annotate: str | None = None,
                  annotate_size: int = 9,
                  ) -> Layer:
        layer = Layer(self._fig,
                      ids=filter_by_name,
                      marker=marker,
                      color=color,
                      marker_size=marker_size,
                      size_extreme=size_extreme,
                      zorder=zorder,
                      images=images,
                      image_size=image_size,
                      annotate=annotate,
                      annotate_size=annotate_size,
                      )

        if store:
            self.layers.append(layer)

        return layer

    def plot(self):
        for layer in self.layers:
            layer.plot(self.db_subset, self.horizontal, self.vertical)

        if self.ss_layer is not None:
            self.ss_layer.plot(self.db_ss, self.horizontal, self.vertical)

    ######################
    #
    # Closing up
    #
    ######################

    def save(self, filename, postfix=".png"):
        filename = self.out_dir/filename
        if not len(Path(filename).suffixes):
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
