import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def plot_multi_time_traces(datasets, dataset_labels, spectral_values, align_datasets=False, xlim=None, ylim=None):
    """
    Plots data from one or more datasets at multiple spectral values.

    For each spectral value, it plots 'fitted_data' as a line and raw 'data'
    as corresponding scatter points. Can optionally align datasets in time.

    Args:
        datasets (xr.Dataset or list): A single dataset or a list of datasets.
        dataset_labels (str or list): A label or list of labels for the datasets.
        spectral_values (float or list): A single spectral value or a list of
                                         values to plot (e.g., [450, 550.5]).
                                         Finds the nearest index automatically.
        align_datasets (bool): If True, each dataset's time axis is shifted
                               so the peak of its 'fitted_data' is at t=0.
                               Defaults to False.
    """
    # --- 1. Standardize inputs to be lists for easier iteration ---
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(dataset_labels, list):
        dataset_labels = [dataset_labels]
    if not isinstance(spectral_values, list):
        spectral_values = [spectral_values]

    if len(datasets) != len(dataset_labels):
        raise ValueError("The number of datasets must match the number of labels.")

    # --- 2. Create the plot ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # --- 3. Iterate through each dataset and each spectral value ---
    for ds, ds_label in zip(datasets, dataset_labels):
        
        # --- 4: Handle Time Alignment ---
                  
        time_coords_for_plot = ds['time']
        
        if align_datasets:
            try:
                irf_offset = ds['irf_center'].item()
            except (KeyError, AttributeError):
                print(f"Warning: 'irf_center' not found in '{ds_label}'. Assuming offset is 0.")
                irf_offset = 0
            
            # align based on irf_center
            time_coords_for_plot = ds['time'] - irf_offset

        for spec_val in spectral_values:
            try:
                # Select data using the nearest value
                data_slice = ds['data'].sel(spectral=spec_val, method='nearest')
                fitted_slice = ds['fitted_data'].sel(spectral=spec_val, method='nearest')
                
                actual_spec_val = fitted_slice['spectral'].item()
                legend_label = f"{ds_label} ({actual_spec_val:.1f} nm)"

                # Plot the fitted data line and get its color
                line, = ax.plot(time_coords_for_plot, fitted_slice, label=legend_label, linewidth=2)
                
                # Plot the raw data with the same color
                ax.scatter(time_coords_for_plot, data_slice, color=line.get_color(), 
                           alpha=0.15, s=10, zorder=-1)

            except (KeyError, IndexError) as e:
                print(f"Could not plot for {ds_label} at {spec_val}: {e}")

    # --- 4. Final plot formatting ---
    xlabel = "Time relative to maximum (ps)" if align_datasets else "Time (ps)"
    ax.set_title("Data vs. Fit for Multiple Datasets and Wavelengths")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Data Value (ΔA)")
    ax.legend(title="Dataset (Wavelength)", fontsize='medium')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    
    # Add a vertical line at t=0 if aligned
    if align_datasets:
        ax.axvline(0, color='grey', linestyle='--', linewidth=1)
    
    # Set axis limits if provided ---
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
        
    plt.tight_layout()
    plt.show()