import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from scipy.signal import savgol_filter


def plot_multi_spectral_slices(datasets, dataset_labels, time_values,
                               measurement_type="TA", normalize=False, apply_chirp_correction=False,
                               xlim=None, ylim=None, smoothing=False, sg_window = 5, sg_order = 0):
    """
    Plots spectral slices with specific logic for TA or TRPL measurements.

    Args:
        datasets (list): A list of datasets.
        dataset_labels (list): A list of labels for the datasets.
        time_values (list): Time values to plot. Interpretation depends on measurement_type.
        measurement_type (str): "TA" or "TRPL". Determines how time-zero is defined.
        apply_chirp_correction (bool): If True and type is "TA", applies a
                                       spectrally-dependent time shift.
        xlim (tuple, optional): A tuple (min, max) for the x-axis limits.
        ylim (tuple, optional): A tuple (min, max) for the y-axis limits.
    """
    # --- 1. Validate inputs and set up plot ---
    if measurement_type not in ["TA", "TRPL"]:
        raise ValueError("measurement_type must be either 'TA' or 'TRPL'.")
    if not isinstance(datasets, list): datasets = [datasets]
    if not isinstance(dataset_labels, list): dataset_labels = [dataset_labels]
    if not isinstance(time_values, list): time_values = [time_values]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    num_time_vals = len(time_values)

    # --- 2. Iterate through datasets and time values ---
    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        base_color = colors[i % len(colors)]
        
        try:
            irf_width_offset = ds['irf_width'].item()
        except KeyError:
            print(f"Warning: 'irf_width' not found in '{ds_label}'. Assuming width offset is 0.")
            irf_width_offset = 0
        
        for j, relative_time in enumerate(time_values):
            # --- 3. Determine selection time based on measurement_type ---
            try:
                # --- TA Logic ---
                if measurement_type == "TA":
                    if apply_chirp_correction:
                        try:
                            chirp_data = ds['irf_center_location']
                            # UPDATED: Added - irf_width_offset
                            absolute_times_to_select = relative_time + chirp_data - irf_width_offset
                            data_slice = ds['data'].sel(time=absolute_times_to_select, method='nearest').squeeze()
                            fitted_slice = ds['fitted_data'].sel(time=absolute_times_to_select, method='nearest').squeeze()
                        except KeyError:
                            print(f"Warning: 'irf_center_location' not found in '{ds_label}'. Cannot apply chirp correction.")
                            continue
                    else:
                        # Unchanged: No offset for TA without chirp correction
                        absolute_time_to_select = relative_time
                        data_slice = ds['data'].sel(time=absolute_time_to_select, method='nearest').squeeze()
                        fitted_slice = ds['fitted_data'].sel(time=absolute_time_to_select, method='nearest').squeeze()

                # --- TRPL Logic ---
                elif measurement_type == "TRPL":
                    try:
                        irf_offset = ds['irf_center'].item()
                        # UPDATED: Added - irf_width_offset
                        absolute_time_to_select = relative_time + irf_offset - irf_width_offset
                        data_slice = ds['data'].sel(time=absolute_time_to_select, method='nearest').squeeze()
                        fitted_slice = ds['fitted_data'].sel(time=absolute_time_to_select, method='nearest').squeeze()
                    except KeyError:
                        print(f"Warning: 'irf_center' not found in '{ds_label}' for TRPL mode. Assuming offset is 0.")
                        absolute_time_to_select = relative_time - irf_width_offset
                        data_slice = ds['data'].sel(time=absolute_time_to_select, method='nearest').squeeze()
                        fitted_slice = ds['fitted_data'].sel(time=absolute_time_to_select, method='nearest').squeeze()

                # option for normalization
                if normalize:
                    # Find the value with the maximum absolute magnitude from the fit
                    np_fitted = ds['data'].values
                    if np_fitted.size > 0:
                        norm_val = np_fitted[np.abs(np_fitted).argmax()]
                        if norm_val != 0: # Avoid division by zero
                            data_slice = data_slice / norm_val
                            fitted_slice = fitted_slice / norm_val
                
                if smoothing:
                    fitted_slice = savgol_filter(data_slice, window_length=sg_window, polyorder=sg_order) 

                # --- 4. Plotting logic (common for both types) ---
                lightness_factor = 1.0
                if num_time_vals > 1:
                    lightness_factor = 0.7 + (j / (num_time_vals - 1)) * 0.6
                h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                plot_color = colorsys.hls_to_rgb(h, max(0, min(1, l * lightness_factor)), s)

                legend_label = f"{ds_label} (t={relative_time:.1f} ps)"
                line, = ax.plot(ds['spectral'], fitted_slice, label=legend_label, color=plot_color, linewidth=2)
                ax.scatter(ds['spectral'], data_slice, color=line.get_color(), alpha=0.5, s=10, zorder=-1)

            except Exception as e:
                print(f"Could not plot for {ds_label} at time {relative_time}: {e}")

    # --- 5. Final plot formatting ---
    title = f"Spectral Slices ({measurement_type} Mode)"
    legend_title = "Dataset (Time)"
    if measurement_type == "TA":
        ax.set_ylabel("ΔA (mOD)")
        if apply_chirp_correction:
            title = "Chirp-Corrected Spectral Slices (TA Mode)"
            legend_title = "Dataset (Time relative to t₀)"
    elif measurement_type == "TRPL":
        ax.set_ylabel("I (A.U.)")
        legend_title = "Dataset (Time relative to IRF)"
        
    #ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.legend(title=legend_title)
    #ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    plt.tight_layout()
    #format_publication_plot_no_latex(ax=ax)
    plt.show()