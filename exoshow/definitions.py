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
default_title = {'any_mass': ((r"Mass ($M_J$)", 'log'),
                              (r"Mass ($M_\oplus$)", lambda x: np.array(x) * c.M_jup / c.M_earth)
                              ),
                 'any_a': (("Semi-major axis (AU)", 'log'),
                           (r"Period around 1 $M_{\odot}$ star (days)",
                            lambda x: (np.array(x) ** 1.5) * 365.24)
                           ),
                 'planet_temp': (("Temperature (K)", None),
                                 ),
                 'planet_flux': ((r"Flux received ($F_\oplus$)", 'log'),
                                 ("Equilibrium Temperature (K)",
                                  lambda x: (x * (c.L_sun / c.au / c.au / c.sigma_sb).value / 16 / np.pi) ** 0.25)
                                 ),
                 'transit_duration': (('Approximate transit duration (h)', None),
                                      ),
                 'planet_gravity': ((r'Planet Gravity ($m/s^{2}$', 'log'),
                                    ("Compared to Earth", lambda x: x / 9.7)
                                    ),
                 'transit_modulation': ((r'$\lambda-to-\lambda$ transit modulation', 'linear'),
                                        ),
                 'transit_snr': (('Expected SNR (arbitrary scale)', 'linear'),
                                 )
                 }
