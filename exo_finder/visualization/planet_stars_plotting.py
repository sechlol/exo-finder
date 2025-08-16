from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt


def plot_planet_period_radius_mass(period_mass: np.ndarray, period_radius: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    ax1.scatter(np.log10(period_mass[:, 0]), period_mass[:, 1], s=1)
    ax2.scatter(np.log10(period_radius[:, 0]), period_radius[:, 1], s=1)
    ax1.set(
        xlabel="Orbital Period (log10 d)",
        ylabel="Planet Mass (M_earth)",
        yscale="log",
        title="Orbital Period vs Mass",
        # xlim=[None, 5],
        # ylim=(0.1, None),
    )
    ax2.set(
        xlabel="Orbital Period (log10 d)",
        ylabel="Planet Radius (R_earth)",
        yscale="log",
        title="Orbital Period vs Radius",
        # xlim=(None, 3.5),
    )

    plt.tight_layout()
    plt.show()


def plot_transit_depth(pl_radius: Sequence[float], transit_depth: Sequence[float], stellar_radius: Sequence[float]):
    # Create a scatter plot with st_rad as the color dimension
    scatter = plt.scatter(pl_radius, transit_depth, c=stellar_radius, s=5, alpha=0.7, cmap="jet")

    plt.xlabel("Planet Radius (Earth radii)")
    plt.ylabel("Transit Depth %")

    plt.ylim(0, 3.5)
    plt.title("Planet Radius vs Transit Depth (colored by Stellar Radius)")

    # Add a color bar to show the stellar radius scale
    cbar = plt.colorbar(scatter)
    cbar.set_label("Stellar Radius (Solar radii)")

    plt.show()
