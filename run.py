from pathlib import Path

import exoplaneteu as eu

ma = eu.MassAxis()
ma.show()

ma.reset_figure()
ma.plot_data(marker="x",
             color="purple",
             marker_size=60,
             xlim=[9.87999124e-05, 5.51703404e+04],
             ylim=[3.13091037e-05, 1.50109695e+02],
             )

ma.props['show_legend'] = False
ma.set_method("transit")
filename = Path(f"{ma.props['db_date']}/transit.png")
ma.save(filename)
ma.props['show_legend'] = True


#
# for year in range(1987, 2024):
#     filename = Path(f"{ma.props['db_date']}/before/{year}.png")
#
#     ma.set_before(year, reset=True)
#     ma.save(filename)
#
#
# year = 2024
# ma.reset_subset()
# filename = Path(f"{ma.props['db_date']}/before/{year}.png")
# ma.save(filename)
#

mat = eu.MassAxis(show_molecules=True)
# ma.plot_data(marker="detection_type", color="white",
#              marker_size=60)

mat.reset_figure()
mat.plot_data(marker="detection_type",
             color="black",
             marker_size=80,
             xlim=[9.87999124e-05, 5.51703404e+04],
             ylim=[3.13091037e-05, 1.50109695e+02],
             )



mat.reset_subset()
filename = Path(f"{mat.props['db_date']}/molecules.png")
mat.save(filename)
