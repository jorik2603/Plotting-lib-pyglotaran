import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def plot_fit_vs_data(dataset, spectral_index):
    """
    Plots the raw data vs the fitted data for a specific spectral index.

    Useful for quickly inspecting the quality of a fit at a specific wavelength.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing 'data', 'fitted_data', and 'spectral' coordinates.
    spectral_index : int
        The integer index along the spectral dimension to plot.

    Returns
    -------
    None
        Displays the plot.
    """
    try:
        data_slice = dataset['data'].isel(spectral=spectral_index)
        fitted_slice = dataset['fitted_data'].isel(spectral=spectral_index)
        time_coords = dataset['time']
        spectral_value = dataset['spectral'].isel(spectral=spectral_index).item()
    except (KeyError, IndexError) as e:
        print(f"Error selecting data: {e}. Ensure dataset has required variables and correct index.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot fitted data (Line)
    ax.plot(time_coords, fitted_slice, color='orangered', linestyle='-', 
            label='Fitted Data', zorder=10)

    # Plot raw data (Scatter)
    ax.scatter(time_coords, data_slice, alpha=0.6, s=15, 
               label='Raw Data', color='steelblue')

    ax.set_xlabel("Time")
    ax.set_ylabel("Data Value")
    ax.set_title(f"Data vs. Fit at Spectral Index {spectral_index} (Wavelength ≈ {spectral_value:.1f})")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()