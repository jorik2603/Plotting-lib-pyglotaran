import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from scipy.signal import savgol_filter

def plot_multi_time_traces(datasets, dataset_labels, spectral_values,
                           measurement_type="TA", normalize=False, apply_chirp_correction=False,
                           xlim=None, ylim=None, smoothing=False, sg_window=5, sg_order=0):
    """
    Plots kinetic time traces for multiple datasets at specific wavelengths.

    Parameters
    ----------
    datasets : list or xarray.Dataset
        The dataset(s) to plot.
    dataset_labels : list or str
        Labels for the datasets.
    spectral_values : list or float
        The specific wavelengths (in nm) to trace over time.
    measurement_type : str, optional
        "TA" (Transient Absorption) or "TRPL". Default is "TA".
    normalize : bool, optional
        If True, normalize traces to peak max. Default is False.
    apply_chirp_correction : bool, optional
        (TA only) If True, corrects the time axis for the IRF center location at that wavelength.
        Default is False.
    xlim, ylim : tuple, optional
        Limits for x (time) and y (amplitude) axes.
    smoothing : bool, optional
        Apply Savitzky-Golay smoothing. Default is False.
    sg_window : int, optional
        Smoothing window length.
    sg_order : int, optional
        Smoothing polynomial order.

    Returns
    -------
    None
        Displays the plot.
    """
    # Validation
    if measurement_type not in ["TA", "TRPL"]:
        raise ValueError("measurement_type must be either 'TA' or 'TRPL'.")
    if not isinstance(datasets, list): datasets = [datasets]
    if not isinstance(dataset_labels, list): dataset_labels = [dataset_labels]
    if not isinstance(spectral_values, list): spectral_values = [spectral_values]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    num_spec_vals = len(spectral_values)

    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        base_color = colors[i % len(colors)]
        
        # Determine IRF offsets
        try:
            irf_width_offset = ds['irf_width'].item()
        except KeyError:
            irf_width_offset = 0
            
        try:
            irf_center_offset = ds['irf_center'].item()
        except KeyError:
            irf_center_offset = 0

        for j, spec_val in enumerate(spectral_values):
            try:
                time_coords_base = ds['time'].copy()

                # Calculate Time Axis based on measurement type and corrections
                if measurement_type == "TA":
                    if apply_chirp_correction:
                        try:
                            # Select nearest chirp value for this wavelength
                            chirp_offset = ds['irf_center_location'].sel(spectral=spec_val, method='nearest').item()
                            # Correction: Shift by chirp + width
                            time_coords_for_plot = time_coords_base - chirp_offset + irf_width_offset
                        except KeyError:
                            print(f"Warning: 'irf_center_location' missing in '{ds_label}'. Skipping chirp correction.")
                            continue
                    else:
                        time_coords_for_plot = time_coords_base

                elif measurement_type == "TRPL":
                    time_coords_for_plot = time_coords_base - irf_center_offset + irf_width_offset
                    
                # Color Variation for multiple wavelengths per dataset
                lightness_factor = 1.0
                if num_spec_vals > 1:
                    lightness_factor = 0.7 + (j / (num_spec_vals - 1)) * 0.6
                h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                plot_color = colorsys.hls_to_rgb(h, max(0, min(1, l * lightness_factor)), s)

                # Select Data
                data_slice = ds['data'].sel(spectral=spec_val, method='nearest')
                fitted_slice = ds['fitted_data'].sel(spectral=spec_val, method='nearest')
                actual_spec_val = fitted_slice['spectral'].item()
                legend_label = f"{ds_label} ({actual_spec_val:.1f} nm)"
                
                # Normalization
                if normalize:
                    np_fitted = fitted_slice.values
                    if np_fitted.size > 0:
                        norm_val = np_fitted[np.abs(np_fitted).argmax()]
                        if norm_val != 0:
                            data_slice = data_slice / norm_val
                            fitted_slice = fitted_slice / norm_val
                            
                # Smoothing
                if smoothing:
                    fitted_slice = savgol_filter(fitted_slice, window_length=sg_window, polyorder=sg_order) 
                    
                # Plot
                line, = ax.plot(time_coords_for_plot, fitted_slice, label=legend_label, color=plot_color, linewidth=2)
                ax.scatter(time_coords_for_plot, data_slice, color=line.get_color(), alpha=0.5, s=10, zorder=-1)

            except Exception as e:
                print(f"Could not plot time trace for {ds_label} at {spec_val}: {e}")
    
    # Formatting
    xlabel = "Time (ps)"
    ylabel = "ΔA (mOD)" if measurement_type == "TA" else "I (A.U.)"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Dataset (Wavelength)")
    ax.axhline(0, color='black', linewidth=0.5)
    
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    plt.tight_layout()
    plt.show()