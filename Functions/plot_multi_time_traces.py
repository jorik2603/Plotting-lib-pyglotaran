import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

def plot_multi_time_traces(datasets, dataset_labels, spectral_values,
                           measurement_type="TA", apply_chirp_correction=False,
                           xlim=None, ylim=None):
    """
    Plots time traces with specific time-zero logic for TA or TRPL measurements.

    Args:
        datasets (list): A list of datasets.
        dataset_labels (list): A list of labels for the datasets.
        spectral_values (list): Spectral values to plot.
        measurement_type (str): "TA" or "TRPL". Determines how time-zero is defined.
        apply_chirp_correction (bool): If True and type is "TA", applies a
                                       spectrally-dependent time shift.
        xlim (tuple, optional): A tuple (min, max) for the x-axis limits.
        ylim (tuple, optional): A tuple (min, max) for the y-axis limits.
    """
    if measurement_type not in ["TA", "TRPL"]:
        raise ValueError("measurement_type must be either 'TA' or 'TRPL'.")
    if not isinstance(datasets, list): datasets = [datasets]
    if not isinstance(dataset_labels, list): dataset_labels = [dataset_labels]
    if not isinstance(spectral_values, list): spectral_values = [spectral_values]

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    num_spec_vals = len(spectral_values)

    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        base_color = colors[i % len(colors)]
        
        # Get offsets for this dataset, with fallbacks
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
                # --- NEW: Time axis calculation logic ---
                time_coords_base = ds['time'].copy()

                # --- TA Logic ---
                if measurement_type == "TA":
                    if apply_chirp_correction:
                        try:
                            chirp_offset = ds['irf_center_location'].sel(spectral=spec_val, method='nearest').item()
                            # Time-zero is chirp + irf_width
                            time_coords_for_plot = time_coords_base - chirp_offset + irf_width_offset
                        except KeyError:
                            print(f"Warning: 'irf_center_location' not found in '{ds_label}'. Cannot apply chirp correction.")
                            continue
                    else:
                        # For TA, time axis is absolute unless corrected
                        time_coords_for_plot = time_coords_base

                # --- TRPL Logic ---
                elif measurement_type == "TRPL":
                    # Time-zero is irf_center + irf_width
                    time_coords_for_plot = time_coords_base - irf_center_offset + irf_width_offset

                # Modify color lightness for each trace
                lightness_factor = 1.0
                if num_spec_vals > 1:
                    lightness_factor = 0.7 + (j / (num_spec_vals - 1)) * 0.6
                h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                plot_color = colorsys.hls_to_rgb(h, max(0, min(1, l * lightness_factor)), s)

                # Select data and plot
                data_slice = ds['data'].sel(spectral=spec_val, method='nearest')
                fitted_slice = ds['fitted_data'].sel(spectral=spec_val, method='nearest')
                actual_spec_val = fitted_slice['spectral'].item()
                legend_label = f"{ds_label} ({actual_spec_val:.1f} nm)"

                line, = ax.plot(time_coords_for_plot, fitted_slice, label=legend_label, color=plot_color, linewidth=2)
                ax.scatter(time_coords_for_plot, data_slice, color=line.get_color(), alpha=0.15, s=10, zorder=-1)

            except Exception as e:
                print(f"Could not plot for {ds_label} at {spec_val}: {e}")

    # Final plot formatting
    xlabel = "Time (ps)"
    if measurement_type == "TA" and apply_chirp_correction:
        xlabel = "Chirp-Corrected Time (ps)"
    elif measurement_type == "TRPL":
        xlabel = "Time relative to IRF (ps)"

    ax.set_title(f"Time Traces at Specific Wavelengths ({measurement_type} Mode)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Data Value (ΔA or Intensity)")
    ax.legend(title="Dataset (Wavelength)", fontsize='medium')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()
  