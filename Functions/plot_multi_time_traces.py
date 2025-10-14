import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

def plot_multi_time_traces(datasets, dataset_labels, spectral_values, 
                           align_datasets=False, apply_chirp_correction=False,
                           xlim=None, ylim=None):
    """
    Plots time traces from datasets at multiple spectral values.

    Can optionally align datasets to their global maximum and apply a
    spectrally-dependent chirp correction to the time axis.

    Args:
        datasets (list): A list of datasets.
        dataset_labels (list): A list of labels for the datasets.
        spectral_values (list): Spectral values to plot.
        align_datasets (bool): If True, shifts each dataset's time axis so the
                               peak of its 'fitted_data' is at t=0.
        apply_chirp_correction (bool): If True, shifts the time axis for each
                                       trace by the value in 'irf_center_location'.
        xlim (tuple, optional): A tuple (min, max) for the x-axis limits.
        ylim (tuple, optional): A tuple (min, max) for the y-axis limits.
    """
    if not isinstance(datasets, list): datasets = [datasets]
    if not isinstance(dataset_labels, list): dataset_labels = [dataset_labels]
    if not isinstance(spectral_values, list): spectral_values = [spectral_values]

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    num_spec_vals = len(spectral_values)

    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        base_color = colors[i % len(colors)]
        time_coords_base = ds['time']
        global_offset = 0

        if align_datasets:
            max_indices = ds['fitted_data'].argmax(dim=['time', 'spectral'])
            global_offset = ds['time'].isel(time=max_indices['time']).item()

        for j, spec_val in enumerate(spectral_values):
            try:
                # Modify color lightness for each trace
                lightness_factor = 1.0
                if num_spec_vals > 1:
                    # Map index j to a lightness multiplier (e.g., from 0.7 to 1.3)
                    lightness_factor = 0.7 + (j / (num_spec_vals - 1)) * 0.6
                
                h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                plot_color = colorsys.hls_to_rgb(h, max(0, min(1, l * lightness_factor)), s)

                chirp_offset = 0
                if apply_chirp_correction:
                    try:
                        chirp_offset = ds['irf_center_location'].sel(spectral=spec_val, method='nearest').item()
                    except KeyError:
                        print(f"Warning: 'irf_center_location' not found in '{ds_label}'. Skipping chirp correction.")
                
                time_coords_for_plot = time_coords_base - global_offset - chirp_offset
                data_slice = ds['data'].sel(spectral=spec_val, method='nearest')
                fitted_slice = ds['fitted_data'].sel(spectral=spec_val, method='nearest')
                actual_spec_val = fitted_slice['spectral'].item()
                legend_label = f"{ds_label} ({actual_spec_val:.1f} nm)"
                
                line, = ax.plot(time_coords_for_plot, fitted_slice, label=legend_label, color=plot_color, linewidth=2)
                ax.scatter(time_coords_for_plot, data_slice, color=line.get_color(), alpha=0.15, s=10, zorder=-1)

            except Exception as e:
                print(f"Could not plot for {ds_label} at {spec_val}: {e}")

    xlabel = "Time (ps)"
    if align_datasets and apply_chirp_correction: xlabel = "Corrected Time relative to maximum (ps)"
    elif align_datasets: xlabel = "Time relative to maximum (ps)"
    elif apply_chirp_correction: xlabel = "Chirp-Corrected Time (ps)"

    ax.set_title("Time Traces at Specific Wavelengths")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Data Value (ΔA)")
    ax.legend(title="Dataset (Wavelength)", fontsize='medium')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    if align_datasets: ax.axvline(0, color='grey', linestyle='--', linewidth=1)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()
