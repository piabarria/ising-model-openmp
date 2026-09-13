from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SERIAL_DATA = ROOT / "results" / "data" / "serial"
OPENMP_DATA = ROOT / "results" / "data" / "openmp"

FIGURES = ROOT / "results" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)


# ============================================================
# Physical reference values
# ============================================================

# Exact critical temperature for the 2D square-lattice Ising model
# with J = k_B = 1:
#
#     T_c = 2 / ln(1 + sqrt(2))
#
TC_ONSAGER = 2.0 / np.log(1.0 + np.sqrt(2.0))


# ============================================================
# Utility functions
# ============================================================

def load_dataset(implementation: str, filename: str) -> pd.DataFrame:
    """
    Load one simulation output file.

    Parameters
    ----------
    implementation
        Either "serial" or "openmp".

    filename
        CSV file name.

    Returns
    -------
    pandas.DataFrame
        Loaded simulation data.
    """

    if implementation == "serial":
        path = SERIAL_DATA / filename

    elif implementation == "openmp":
        path = OPENMP_DATA / filename

    else:
        raise ValueError(
            "implementation must be 'serial' or 'openmp'"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing simulation output: {path}"
        )

    return pd.read_csv(path)


def exponential_correlation(r, amplitude, xi):
    """
    Exponential model for the connected correlation function:

        C(r) = A exp(-r / xi)

    where xi is the correlation length.
    """

    return amplitude * np.exp(-r / xi)


def estimate_correlation_length(
    correlation_row: pd.Series,
    r_values: np.ndarray,
) -> float:
    """
    Estimate the correlation length xi by fitting the positive
    part of the connected spatial correlation function.
    """

    correlation_values = (
        correlation_row
        .to_numpy(dtype=float)
    )

    valid = (
        np.isfinite(correlation_values)
        & (correlation_values > 0.0)
    )

    r_fit = r_values[valid]
    c_fit = correlation_values[valid]

    if len(r_fit) < 3:
        return np.nan

    try:

        parameters, _ = curve_fit(
            exponential_correlation,
            r_fit,
            c_fit,
            p0=(c_fit[0], 1.0),
            bounds=(
                (0.0, 1e-8),
                (np.inf, np.inf),
            ),
            maxfev=10000,
        )

        _, xi = parameters

        return xi

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
    ):
        return np.nan


# ============================================================
# Equilibration
# ============================================================

def plot_equilibration():
    """
    Compare the magnetization equilibration histories of the
    serial and OpenMP implementations.
    """

    serial = load_dataset(
        "serial",
        "equilibration.csv",
    )

    openmp = load_dataset(
        "openmp",
        "equilibration.csv",
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        serial["sweep"],
        serial["abs_magnetization_per_spin"],
        linewidth=1.5,
        label="Serial",
    )

    plt.plot(
        openmp["sweep"],
        openmp["abs_magnetization_per_spin"],
        linewidth=1.5,
        label="OpenMP",
        alpha=0.8,
    )

    plt.xlabel("Monte Carlo sweep")

    plt.ylabel(
        r"$\langle |M| \rangle / N$"
    )

    plt.title(
        "2D Ising Model: Equilibration at T = 2.0"
    )

    plt.ylim(
        -0.05,
        1.05,
    )

    plt.legend()

    plt.tight_layout()

    output = (
        FIGURES
        / "equilibration_magnetization.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# Energy
# ============================================================

def plot_energy():
    """
    Plot the average energy per spin as a function of temperature.
    """

    serial = load_dataset(
        "serial",
        "thermodynamics.csv",
    )

    openmp = load_dataset(
        "openmp",
        "thermodynamics.csv",
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        serial["temperature"],
        serial["energy_per_spin"],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label="Serial",
    )

    plt.plot(
        openmp["temperature"],
        openmp["energy_per_spin"],
        marker="s",
        markersize=3,
        linewidth=1.3,
        label="OpenMP",
    )

    plt.axvline(
        TC_ONSAGER,
        linestyle="--",
        linewidth=1.2,
        label=r"$T_c$ (Onsager)",
    )

    plt.xlabel("Temperature")

    plt.ylabel(
        r"$\langle E \rangle / N$"
    )

    plt.title(
        "Energy per Spin"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        FIGURES
        / "energy_vs_temperature.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# Magnetization
# ============================================================

def plot_magnetization():
    """
    Plot the absolute magnetization per spin.
    """

    serial = load_dataset(
        "serial",
        "thermodynamics.csv",
    )

    openmp = load_dataset(
        "openmp",
        "thermodynamics.csv",
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        serial["temperature"],
        serial[
            "abs_magnetization_per_spin"
        ],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label="Serial",
    )

    plt.plot(
        openmp["temperature"],
        openmp[
            "abs_magnetization_per_spin"
        ],
        marker="s",
        markersize=3,
        linewidth=1.3,
        label="OpenMP",
    )

    plt.axvline(
        TC_ONSAGER,
        linestyle="--",
        linewidth=1.2,
        label=r"$T_c$ (Onsager)",
    )

    plt.xlabel("Temperature")

    plt.ylabel(
        r"$\langle |M| \rangle / N$"
    )

    plt.title(
        "Magnetization per Spin"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        FIGURES
        / "magnetization_vs_temperature.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# Heat capacity
# ============================================================

def plot_heat_capacity():
    """
    Plot the heat capacity per spin.
    """

    serial = load_dataset(
        "serial",
        "thermodynamics.csv",
    )

    openmp = load_dataset(
        "openmp",
        "thermodynamics.csv",
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        serial["temperature"],
        serial["heat_capacity"],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label="Serial",
    )

    plt.plot(
        openmp["temperature"],
        openmp["heat_capacity"],
        marker="s",
        markersize=3,
        linewidth=1.3,
        label="OpenMP",
    )

    plt.axvline(
        TC_ONSAGER,
        linestyle="--",
        linewidth=1.2,
        label=r"$T_c$ (Onsager)",
    )

    plt.xlabel("Temperature")

    plt.ylabel(
        r"$C_V / N$"
    )

    plt.title(
        "Heat Capacity"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        FIGURES
        / "heat_capacity.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# Magnetic susceptibility
# ============================================================

def plot_susceptibility():
    """
    Plot the magnetic susceptibility per spin.
    """

    serial = load_dataset(
        "serial",
        "thermodynamics.csv",
    )

    openmp = load_dataset(
        "openmp",
        "thermodynamics.csv",
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        serial["temperature"],
        serial["susceptibility"],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label="Serial",
    )

    plt.plot(
        openmp["temperature"],
        openmp["susceptibility"],
        marker="s",
        markersize=3,
        linewidth=1.3,
        label="OpenMP",
    )

    plt.axvline(
        TC_ONSAGER,
        linestyle="--",
        linewidth=1.2,
        label=r"$T_c$ (Onsager)",
    )

    plt.xlabel("Temperature")

    plt.ylabel(
        r"$\chi / N$"
    )

    plt.title(
        "Magnetic Susceptibility"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        FIGURES
        / "susceptibility.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# Correlation length
# ============================================================

def compute_correlation_lengths(
    dataframe: pd.DataFrame,
):
    """
    Estimate xi(T) for every temperature contained in a
    correlation dataset.
    """

    correlation_columns = [
        column
        for column in dataframe.columns
        if column.startswith("C_")
    ]

    r_values = np.array(
        [
            int(
                column.split("_")[1]
            )
            for column
            in correlation_columns
        ],
        dtype=float,
    )

    xi_values = []

    for _, row in dataframe.iterrows():

        xi = estimate_correlation_length(
            row[correlation_columns],
            r_values,
        )

        xi_values.append(
            xi
        )

    return np.asarray(
        xi_values
    )


def plot_correlation_length():
    """
    Estimate and compare the temperature dependence of the
    correlation length.
    """

    serial = load_dataset(
        "serial",
        "correlation.csv",
    )

    openmp = load_dataset(
        "openmp",
        "correlation.csv",
    )

    serial_xi = (
        compute_correlation_lengths(
            serial
        )
    )

    openmp_xi = (
        compute_correlation_lengths(
            openmp
        )
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        serial["temperature"],
        serial_xi,
        marker="o",
        markersize=4,
        linewidth=1.3,
        label="Serial",
    )

    plt.plot(
        openmp["temperature"],
        openmp_xi,
        marker="s",
        markersize=4,
        linewidth=1.3,
        label="OpenMP",
    )

    plt.axvline(
        TC_ONSAGER,
        linestyle="--",
        linewidth=1.2,
        label=r"$T_c$ (Onsager)",
    )

    plt.xlabel("Temperature")

    plt.ylabel(
        r"Correlation length $\xi$"
    )

    plt.title(
        "Estimated Correlation Length"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        FIGURES
        / "correlation_length.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Generating Ising-model figures..."
    )

    print(
        f"Exact critical temperature: "
        f"T_c = {TC_ONSAGER:.6f}"
    )

    plot_equilibration()

    plot_energy()

    plot_magnetization()

    plot_heat_capacity()

    plot_susceptibility()

    plot_correlation_length()

    print(
        "All figures generated successfully."
    )


if __name__ == "__main__":
    main()
