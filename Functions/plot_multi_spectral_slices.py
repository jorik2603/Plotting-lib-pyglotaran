import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def plot_multi_spectral_slices(datasets, dataset_labels, time_values, xlim=None, ylim=None):
    """
    Plots spectral slices, offsetting time values by 'irf_center'.

    For each specified relative time, it adds the dataset's 'irf_center'
    value to find the absolute time, then plots the 'fitted_data' as a line
    and 'data' as scatter points against the spectral axis.

    Args:
        datasets (xr.Dataset or list): A single dataset or a list of datasets.
                                     Each dataset should contain a scalar
                                     data variable named 'irf_center'.
        dataset_labels (str or list): A label or list of labels for the datasets.
        time_values (float or list): A single time value or a list of values
                                     relative to irf_center (e.g., [0, 100, 500]).
    """
    # --- 1. Standardize inputs to be lists for easier iteration ---
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(dataset_labels, list):
        dataset_labels = [dataset_labels]
    if not isinstance(time_values, list):
        time_values = [time_values]

    if len(datasets) != len(dataset_labels):
        raise ValueError("The number of datasets must match the number of labels.")

    # --- 2. Create the plot ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # --- 3. Iterate through each dataset and each time value ---
    for ds, ds_label in zip(datasets, dataset_labels):
        spectral_coords = ds['spectral']

        # --- NEW: Get the irf_center offset for the current dataset ---
        try:
            irf_offset = ds['irf_center'].item()
        except (KeyError, AttributeError):
            print(f"Warning: 'irf_center' not found in '{ds_label}'. Assuming offset is 0.")
            irf_offset = 0

        for relative_time in time_values:
            try:
                # Calculate the absolute time to select in the dataset's coordinates
                absolute_time_to_select = relative_time + irf_offset

                # Select data using the calculated absolute time
                data_slice = ds['data'].sel(time=absolute_time_to_select, method='nearest')
                fitted_slice = ds['fitted_data'].sel(time=absolute_time_to_select, method='nearest')

                # The legend now shows the user-provided relative time for clarity
                legend_label = f"{ds_label} (t={relative_time:.1f} ps)"

                # Plot the fitted data line and get its color
                line, = ax.plot(spectral_coords, fitted_slice, label=legend_label, linewidth=2)
                
                # Plot the raw data with the same color
                ax.scatter(spectral_coords, data_slice, color=line.get_color(), 
                           alpha=0.15, s=10, zorder=-1)

            except (KeyError, IndexError) as e:
                print(f"Could not plot for {ds_label} at time {relative_time}: {e}")

    # --- 4. Final plot formatting ---
    ax.set_title("Spectral Slices at Time Points Relative to IRF Center")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Data Value (ΔA)")
    ax.legend(title="Dataset (Time relative to IRF)", fontsize='medium')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    
    # --- NEW: Set axis limits if provided ---
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
        
    plt.tight_layout()
    plt.show()