from typing import Optional, Union

import numpy as np
from astropy.time import Time
from astropy.units import Quantity
from lightkurve import LightCurve
from matplotlib import pyplot as plt

from exotools import LightCurvePlus, StarSystem, Planet

_cmap = plt.get_cmap("tab10")


def plot_lightcurve_ax(
    lightcurve: LightCurve | LightCurvePlus,
    ax: plt.Axes,
    label: Optional[str] = None,
    title: Optional[str] = None,
):
    ax.scatter(lightcurve.time.value, lightcurve.flux.value, color="blue", s=1, label=label)
    if isinstance(lightcurve.time, Time):
        ax.set(xlabel=f"Days (format: {lightcurve.time.format}, scale: {lightcurve.time.scale})", ylabel="Flux")
    elif isinstance(lightcurve.time, Quantity):
        ax.set(xlabel=f"Days (normalized)", ylabel="Flux")
    ax.grid(True)
    if title:
        ax.set(title=title)


def plot_lightcurve_and_image(lightcurve: LightCurve, image_data: np.ndarray, title: Optional[str] = None):
    time = lightcurve.time.value
    flux = lightcurve.flux.value

    # Plot the light curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_lightcurve_ax(lightcurve, ax1)
    ax2.imshow(image_data, cmap="viridis")
    ax1.set(title="Light Curve")
    ax2.set(title="Raw Frame")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)


def plot_lightcurve(lightcurve: LightCurve, title: Optional[str] = None):
    # Plot the light curve
    fig, ax = plt.subplots(figsize=(15, 4))
    plot_lightcurve_ax(lightcurve, ax)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)


def plot_lightcurve_folded(
    lightcurve: LightCurve, planet: Planet, title: Optional[str] = None, ax: Optional[plt.Axes] = None
):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 4))
    if title:
        ax.set(title=title)

    folded_lc = lightcurve.fold(period=planet.orbital_period.central, epoch_time=planet.transit_midpoint.central)

    plot_lightcurve_ax(folded_lc, ax=ax, title=title)

    if show:
        plt.tight_layout()
        plt.show(block=False)


def plot_lightcurve_and_fit(
    lightcurve: LightCurvePlus,
    fit: Union[LightCurve, np.ndarray],
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    fit = lightcurve.copy_with_flux(flux=fit) if not isinstance(fit, LightCurvePlus) else fit
    show = ax is None

    # Plot the light curve
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 4))
    plot_lightcurve_ax(lightcurve, ax, label="data")
    ax.plot(fit.time_x, fit.flux_y, color="red", label="fit")
    if title:
        ax.set(title=title)

    if show:
        plt.tight_layout()
        plt.show(block=False)


def plot_lightcurve_masked(
    lightcurve: LightCurvePlus | LightCurve, planet: Planet, ax: Optional[plt.Axes] = None, label: Optional[str] = None
):
    if not isinstance(lightcurve, LightCurvePlus):
        lightcurve = LightCurvePlus(lightcurve)

    min_y = lightcurve.flux_y.min()
    max_y = lightcurve.flux_y.max()
    time_x = lightcurve.time_x
    transit_mask = lightcurve.get_transit_mask(planet)

    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 4))

    ax.fill_between(time_x, min_y, max_y, where=transit_mask, alpha=0.5, interpolate=True, label=planet.name)
    plot_lightcurve_ax(lightcurve=lightcurve, ax=ax, label=label)

    if show:
        plt.tight_layout()
        plt.show(block=False)


def plot_planet_transits(lightcurve: LightCurvePlus | LightCurve, system: StarSystem, ax: Optional[plt.Axes] = None):
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 4))

    plot_lightcurve_ax(lightcurve, ax=ax)
    time_x: np.ndarray = lightcurve.time.value
    min_y: float = lightcurve.flux.min()
    max_y: float = lightcurve.flux.max()

    for i, planet in enumerate(system.planets):
        mask = lightcurve.get_transit_mask(planet)
        ax.fill_between(
            time_x, min_y, max_y, where=mask, alpha=0.5, color=_cmap(i), interpolate=True, label=planet.name
        )

    ax.legend()
    if show:
        plt.tight_layout()
        plt.show(block=False)
