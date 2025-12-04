import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def plot_fit_vs_data(dataset, spectral_index):
    """
    Plots fitted data (line) and raw data (scatter) for a specific spectral index.

    Args:
        dataset (xr.Dataset): The dataset containing the data. It must have 'time'
                              and 'spectral' coordinates, and 'data' and 
                              'fitted_data' variables.
        spectral_index (int): The integer index along the 'spectral' dimension 
                              to plot.
    """
    # --- 1. Select the data for the chosen index ---
    # Extracts the 1D time series for both variables at the specified spectral index.
    try:
        data_slice = dataset['data'].isel(spectral=spectral_index)
        fitted_slice = dataset['fitted_data'].isel(spectral=spectral_index)
        time_coords = dataset['time']
        spectral_value = dataset['spectral'].isel(spectral=spectral_index).item()
    except (KeyError, IndexError) as e:
        print(f"Error selecting data: {e}. Ensure dataset has the required variables/dimensions.")
        return

    # --- 2. Create the plot ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the fitted data as a solid line
    ax.plot(time_coords, fitted_slice, color='orangered', linestyle='-', 
            label='Fitted Data', zorder=10)

    # Plot the raw data as a scatter plot
    ax.scatter(time_coords, data_slice, alpha=0.6, s=15, 
               label='Raw Data', color='steelblue')

    # --- 3. Add labels and title for clarity ---
    ax.set_xlabel("Time")
    ax.set_ylabel("Data Value")
    ax.set_title(f"Data vs. Fit at Spectral Index {spectral_index} (Wavelength ≈ {spectral_value:.1f})")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
