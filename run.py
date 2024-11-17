from pathlib import Path

import exoplaneteu as eu

#ma = eu.MassAxis()
ma = eu.MassAxis()
ma.plot_data(marker="detection_type", color="white",
             marker_size=60)

for year in range(1987, 2024):
    filename = Path(f"{ma.props['db_date']}/before/{year}.png")
    ma.set_before(year)
    ma.plot_data(marker="detection_type",
                 color="black",
                 marker_size=60)
    ma.save(filename)

ma.show()

