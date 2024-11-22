import numpy as np
from astropy import constants as c

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
