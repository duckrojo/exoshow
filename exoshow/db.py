import re
from glob import glob
from pathlib import Path

import astropy.constants as c
import numpy as np
import pandas as pd


def read_exoplanets(directory,
                    date=None,
                    planet_albedo=0.1,
                    t_sun=5777,
                    ):
    filename = sorted(glob(str(directory / 'exoplanet*csv')))[-1]

    rec = re.compile(r"_(\d+)[_.]")
    match = re.search(rec, filename)

    if date is None:
        date = match.group(1)
    elif not (directory / f'exoplanet_{date}.csv').exists():
        raise ValueError(f"exoplanet DB with date '{date}' was not found")

    exop = pd.read_csv(filename, index_col='name')

    # Using Kepler a(P,M)
    exop['semi_major_axis'].fillna(((exop['orbital_period'] / 365.24) ** 2 * exop['star_mass']) ** (-1 / 3))
    # Using equilibrium flux F(T_star,R_star,a) in comparison with Earth and assuming same albedo
    exop['planet_flux'] = (exop['star_teff'] / t_sun) ** 4 * (exop['star_radius'] / exop['semi_major_axis']) ** 2
    # Using equilibrium temperature F(T_star,R_star,a)
    exop['planet_temperature'] = np.sqrt(exop['star_radius'] * c.R_sun / c.au / 2.0 / exop['semi_major_axis'] *
                                         np.sqrt(1 - planet_albedo)) * exop['star_teff']

    exop.loc[:, 'any_mass'] = exop['mass'].copy()
    exop.loc[:, 'any_a'] = exop['semi_major_axis'].copy()
    exop.fillna({'any_a': (exop['star_mass'] * exop['orbital_period'] ** 2) ** (1 / 3),
                 'any_mass': exop['mass_sini'],
                 },
                inplace=True)

    exop['inversions'] = False
    for name in ['WASP-33 Ab', 'WASP-18 Ab', 'WASP-121 b']:
        if name not in exop.index:
            raise ValueError(f"Planet with inversion f{name} not found in this exoplanet "
                             f"version {date}")
        exop.loc[name, 'inversions'] = True

    exop['transit_duration'] = (2 * exop['star_radius'] * c.R_sun /
                                (2 * np.pi * exop['any_a'] * c.au / (exop['orbital_period'] * 24)))

    exop['planet_gravity'] = c.G * exop['any_mass'] * c.M_jup / (exop['radius'] * c.R_jup) ** 2

    exop['planet_density'] = 3 * exop['any_mass'] / (4 * np.pi * exop['radius'] ** 3)

    exop['mean_molecular_mass'] = 2.22 + (29 - 2.22) * (exop['planet_density'] > 3)

    exop['scale_height'] = (c.k_B * exop['planet_temperature'] /
                            (exop['mean_molecular_mass'] * c.m_p * exop['planet_gravity']))

    exop['transit_modulation'] = exop['radius'] * exop['scale_height'] / exop['star_radius'] ** 2

    exop['transit_depth'] = (exop['radius'] * c.R_jup / exop['star_radius'] / c.R_sun) ** 2

    exop['transit_snr'] = exop['transit_modulation'] * 10 ** (-0.2 * exop['mag_v']) * np.sqrt(exop['transit_duration'])

    return date, exop


def read_solar_system(filename=None,
                      planet_albedo=0.3,
                      t_sun=5777,
                      ):

    if filename is None:
        filename = Path(__file__).resolve().parent.parent / 'db/solar_system.csv'

    ss = pd.read_csv(filename, index_col='Name').drop(index="PLUTO")

    # Solar System mass in Jupiter units
    ss["any_mass"] = ss["Mass"] / ss["Mass"].loc["JUPITER"]
    ss['any_a'] = ss['Distance from Sun'].copy()
    ss['planet_temp'] = np.sqrt(c.R_sun / c.au / 2.0 / ss['any_a'] * np.sqrt(1 - planet_albedo)) * t_sun
    ss['planet_flux'] = 1 / ss['any_a'] ** 2

    return ss


def add_column(db, **kwargs):
    for column, content in kwargs.items():
        db[column] = np.nan
        for name, value in content.items():
            db.at[name, column] = value

    return db


def compute(db, **kwargs):
    for column, formula in kwargs.items():
        formula = formula.replace('<index>', 'self.db_exoplanet.index')
        formula = formula.replace('<', 'self.db_exoplanet["')
        formula = formula.replace('>', '"]')
        db[column] = eval(formula)

    return db
