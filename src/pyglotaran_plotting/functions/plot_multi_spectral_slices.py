import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from scipy.signal import savgol_filter

def plot_multi_spectral_slices(datasets, dataset_labels, time_values,
                               measurement_type="TA", normalize=False, apply_chirp_correction=False,
                               xlim=None, ylim=None, smoothing=False, sg_window=5, sg_order=0):
    """
    Plots spectral slices (spectra at specific times) for multiple datasets.

    Handles different logic for Transient Absorption (TA) and Time-Resolved Photoluminescence (TRPL),
    including automatic chirp correction and normalization.

    Parameters
    ----------
    datasets : list or xarray.Dataset
        The dataset(s) containing the data.
    dataset_labels : list or str
        Label(s) for the dataset(s).
    time_values : list or float
        The specific time points (in ps) at which to slice the spectra.
    measurement_type : str, optional
        "TA" or "TRPL". Determines how time-zero and IRF offsets are handled. Default is "TA".
    normalize : bool, optional
        If True, normalizes the slices to the dataset's maximum intensity. Default is False.
    apply_chirp_correction : bool, optional
        (TA only) If True, adjusts the slicing time based on the wavelength-dependent IRF center.
        Default is False.
    xlim, ylim : tuple, optional
        Limits for the x (wavelength) and y (intensity) axes.
    smoothing : bool, optional
        If True, applies Savitzky-Golay smoothing to the data. Default is False.
    sg_window : int, optional
        Window length for smoothing. Default is 5.
    sg_order : int, optional
        Polynomial order for smoothing. Default is 0.

    Returns
    -------
    None
        Displays the plot.
    """
    # Validate inputs
    if measurement_type not in ["TA", "TRPL"]:
        raise ValueError("measurement_type must be either 'TA' or 'TRPL'.")
    if not isinstance(datasets, list): datasets = [datasets]
    if not isinstance(dataset_labels, list): dataset_labels = [dataset_labels]
    if not isinstance(time_values, list): time_values = [time_values]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    num_time_vals = len(time_values)

    # Loop through datasets
    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        base_color = colors[i % len(colors)]
        
        # Get IRF width offset if available
        try:
            irf_width_offset = ds['irf_width'].item()
        except KeyError:
            print(f"Warning: 'irf_width' not found in '{ds_label}'. Assuming width offset is 0.")
            irf_width_offset = 0
        
        # Loop through requested time points
        for j, relative_time in enumerate(time_values):
            try:
                # Logic for determining the absolute time to slice
                if measurement_type == "TA":
                    if apply_chirp_correction:
                        try:
                            chirp_data = ds['irf_center_location']
                            # Calculate time per wavelength
                            absolute_times_to_select = relative_time + chirp_data - irf_width_offset
                            data_slice = ds['data'].sel(time=absolute_times_to_select, method='nearest').squeeze()
                            fitted_slice = ds['fitted_data'].sel(time=absolute_times_to_select, method='nearest').squeeze()
                        except KeyError:
                            print(f"Warning: 'irf_center_location' missing in '{ds_label}'. Skipping chirp correction.")
                            continue
                    else:
                        absolute_time_to_select = relative_time
                        data_slice = ds['data'].sel(time=absolute_time_to_select, method='nearest').squeeze()
                        fitted_slice = ds['fitted_data'].sel(time=absolute_time_to_select, method='nearest').squeeze()

                elif measurement_type == "TRPL":
                    try:
                        irf_offset = ds['irf_center'].item()
                        absolute_time_to_select = relative_time + irf_offset - irf_width_offset
                    except KeyError:
                        print(f"Warning: 'irf_center' missing in '{ds_label}'. Assuming offset is 0.")
                        absolute_time_to_select = relative_time - irf_width_offset
                    
                    data_slice = ds['data'].sel(time=absolute_time_to_select, method='nearest').squeeze()
                    fitted_slice = ds['fitted_data'].sel(time=absolute_time_to_select, method='nearest').squeeze()

                # Normalization
                if normalize:
                    np_data = ds['data'].values
                    if np_data.size > 0:
                        norm_val = np_data[np.abs(np_data).argmax()]
                        if norm_val != 0:
                            data_slice = data_slice / norm_val
                            fitted_slice = fitted_slice / norm_val
                
                # Smoothing
                if smoothing:
                    fitted_slice = savgol_filter(fitted_slice, window_length=sg_window, polyorder=sg_order) 

                # Color logic: varied lightness for different time points
                lightness_factor = 1.0
                if num_time_vals > 1:
                    lightness_factor = 0.7 + (j / (num_time_vals - 1)) * 0.6
                h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                plot_color = colorsys.hls_to_rgb(h, max(0, min(1, l * lightness_factor)), s)

                legend_label = f"{ds_label} (t={relative_time:.1f} ps)"
                
                # Plotting
                line, = ax.plot(ds['spectral'], fitted_slice, label=legend_label, color=plot_color, linewidth=2)
                ax.scatter(ds['spectral'], data_slice, color=line.get_color(), alpha=0.5, s=10, zorder=-1)

            except Exception as e:
                print(f"Could not plot for {ds_label} at time {relative_time}: {e}")

    # Formatting
    legend_title = "Dataset (Time)"
    if measurement_type == "TA":
        ax.set_ylabel("ΔA (mOD)")
        if apply_chirp_correction:
            legend_title = "Dataset (Time relative to t₀)"
    elif measurement_type == "TRPL":
        ax.set_ylabel("I (A.U.)")
        legend_title = "Dataset (Time relative to IRF)"
        
    ax.set_xlabel("Wavelength (nm)")
    ax.legend(title=legend_title)
    ax.axhline(0, color='black', linewidth=0.5)
    
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.show()