from typing import Optional

import numpy as np
from astropy.units import Quantity
from matplotlib import pyplot as plt
from scipy import stats


def fit_distributions_to_data_and_visualize(
    data: Quantity,
    name: str,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
):
    # Exclude 0 values
    data = data.filled(0)
    data = data[data != 0]

    print(f"data mean: {data.mean()}, st: {data.std()}, min-max: [{data.min()}, {data.max()}]")
    if lower_limit is not None:
        data = data[data.value > lower_limit]
    if upper_limit is not None:
        data = data[data.value < upper_limit]

    print(f"filtered_data mean: {data.mean()}, st: {data.std()}, min-max: [{data.min()}, {data.max()}]")
    max_val = data.max()

    # Ensure data is strictly between 0 and 1 for beta distribution
    # Avoid exact 0 and 1 values which can cause issues with beta distribution
    epsilon = 1e-10
    positive_data = data[data >= 0]
    normalized_data = positive_data / max_val
    normalized_data = normalized_data.clip(epsilon, 1.0 - epsilon)

    alpha_fit = stats.alpha.fit(normalized_data)
    beta_fit = stats.beta.fit(normalized_data)
    gamma_fit = stats.gamma.fit(positive_data)
    normal_fit = stats.norm.fit(data)

    x_data = np.linspace(start=data.min(), stop=max_val, num=1000)
    x_positive = np.linspace(start=0, stop=max_val, num=1000)
    x_normalized = np.linspace(start=0, stop=1, num=1000)

    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    axes = axes.flatten()

    axes[0].hist(normalized_data, density=True, bins=100, label="data")
    axes[0].plot(x_normalized, stats.alpha.pdf(x_normalized, *alpha_fit), label="Fit", color="red")
    axes[0].set(title="Alpha Fit", xlabel=f"{data.unit}")

    axes[1].hist(normalized_data, density=True, bins=100, label="data")
    axes[1].plot(x_normalized, stats.beta.pdf(x_normalized, *beta_fit), label="Fit", color="red")
    axes[1].set(title="Beta Fit", xlabel=f"{data.unit}")

    axes[2].hist(positive_data, density=True, bins=100, label="data")
    axes[2].plot(x_positive, stats.gamma.pdf(x_positive, *gamma_fit), label="Fit", color="red")
    axes[2].set(title="Gamma Fit", xlabel=f"{data.unit}")

    axes[3].hist(data, density=True, bins=100, label="data")
    axes[3].plot(x_data, stats.norm.pdf(x_data, *normal_fit), label="Fit", color="red")
    axes[3].set(title="Normal Fit", xlabel=f"{data.unit}")

    plt.suptitle(f"Fit for parameter: {name}")
    plt.tight_layout()
    plt.show()

    print("Alpha parameters:", alpha_fit)
    print("Beta parameters:", beta_fit)
    print("Gamma parameters:", gamma_fit)
    print("Normal parameters:", normal_fit)
