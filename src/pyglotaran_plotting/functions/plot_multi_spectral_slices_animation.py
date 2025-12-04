import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from scipy.signal import savgol_filter

def plot_multi_spectral_slices_animation(datasets, dataset_labels, time_values, ax,
                               measurement_type="TA", normalize=False, apply_chirp_correction=False,
                               xlim=None, ylim=None, smoothing=False, sg_window=5, sg_order=0):
    """
    Plots a single frame of spectral slices for use in an animation loop.

    This function clears the provided axis and redraws the spectral slices for the 
    given datasets and time values.

    Parameters
    ----------
    datasets : list
        List of datasets to plot.
    dataset_labels : list
        Labels for the datasets.
    time_values : list
        List of time values to plot in this specific frame.
    ax : matplotlib.axes.Axes
        The axes object to draw on. This is cleared at the start of the function.
    measurement_type : str, optional
        "TA" or "TRPL". Default is "TA".
    normalize : bool, optional
        Normalize data. Default is False.
    apply_chirp_correction : bool, optional
        (TA only) Apply chirp correction to time selection. Default is False.
    xlim, ylim : tuple, optional
        Axis limits.
    smoothing : bool, optional
        Apply smoothing. Default is False.
    sg_window : int, optional
        Smoothing window.
    sg_order : int, optional
        Smoothing order.

    Returns
    -------
    None
        Modifies the `ax` object in-place.
    """
    # Clear previous frame
    ax.clear()
    
    if measurement_type not in ["TA", "TRPL"]:
        raise ValueError("measurement_type must be either 'TA' or 'TRPL'.")
    if not isinstance(datasets, list): datasets = [datasets]
    if not isinstance(dataset_labels, list): dataset_labels = [dataset_labels]
    if not isinstance(time_values, list): time_values = [time_values]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    num_time_vals = len(time_values)

    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        base_color = colors[i % len(colors)]
        
        try:
            irf_width_offset = ds['irf_width'].item()
        except KeyError:
            irf_width_offset = 0
        
        for j, relative_time in enumerate(time_values):
            try:
                # Time Selection Logic
                if measurement_type == "TA":
                    if apply_chirp_correction:
                        try:
                            chirp_data = ds['irf_center_location']
                            absolute_times_to_select = relative_time + chirp_data - irf_width_offset
                            data_slice = ds['data'].sel(time=absolute_times_to_select, method='nearest').squeeze()
                            fitted_slice = ds['fitted_data'].sel(time=absolute_times_to_select, method='nearest').squeeze()
                        except KeyError:
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
                        absolute_time_to_select = relative_time - irf_width_offset
                        
                    data_slice = ds['data'].sel(time=absolute_time_to_select, method='nearest').squeeze()
                    fitted_slice = ds['fitted_data'].sel(time=absolute_time_to_select, method='nearest').squeeze()

                # Normalization
                if normalize:
                    np_fitted = fitted_slice.values
                    if np_fitted.size > 0:
                        norm_val = np_fitted[np.abs(np_fitted).argmax()]
                        if norm_val != 0:
                            data_slice = data_slice / norm_val
                            fitted_slice = fitted_slice / norm_val

                # Plotting
                lightness_factor = 1.0
                if num_time_vals > 1:
                    lightness_factor = 0.7 + (j / (num_time_vals - 1)) * 0.6
                h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                plot_color = colorsys.hls_to_rgb(h, max(0, min(1, l * lightness_factor)), s)

                legend_label = f"{ds_label} (t={relative_time:.1f} ps)"
                
                if smoothing:
                    fitted_slice = savgol_filter(data_slice, window_length=sg_window, polyorder=sg_order) 
                    
                line, = ax.plot(ds['spectral'], fitted_slice, label=legend_label, color=plot_color, linewidth=2)
                ax.scatter(ds['spectral'], data_slice, color=line.get_color(), alpha=0.5, s=10, zorder=-1)
            except Exception as e:
                print(f"Animation frame error: {ds_label} at {relative_time}: {e}")

    # Labeling
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