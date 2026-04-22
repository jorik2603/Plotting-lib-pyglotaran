import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from brokenaxes import brokenaxes
import colorsys
from pathlib import Path

def plot_multi_spectral_slices(datasets, dataset_labels, time_values,
                               measurement_type="TA", plot_raw = False, apply_chirp_correction=False,legend=True,
                               color=None, normalize = False, xlim=None, ylim=None, broken_axes=False, broken_xlims=None, 
                               broken_width=0.1,export=False,export_folder="slices", return_fig_object=False, hide_spines=False):
    """
    Plots spectral slices with specific logic for TA or TRPL measurements.

    Args:
        datasets (list): A list of datasets.
        dataset_labels (list): A list of labels for the datasets.
        time_values (list): Time values to plot. Interpretation depends on measurement_type.
        measurement_type (str): "TA" or "TRPL". Determines how time-zero is defined.
        apply_chirp_correction (bool): If True and type is "TA", applies a
                                       spectrally-dependent time shift.
        normalize (bool): If True normalizes spectrum now only for TRPL measurement type.
        xlim (tuple, optional): A tuple (min, max) for the x-axis limits.
        ylim (tuple, optional): A tuple (min, max) for the y-axis limits.
        brokenaxes (bool): If True plots broken axes
        broken_xlims ((tuple,tuple),optional): Two tuples (min, max) for the x-axis limits when using brokenaxes.
    """
    # --- 1. Validate inputs and set up plot ---
    if measurement_type not in ["TA", "TRPL"]:
        raise ValueError("measurement_type must be either 'TA' or 'TRPL'.")
    if not isinstance(datasets, list): datasets = [datasets]
    if not isinstance(dataset_labels, list): dataset_labels = [dataset_labels]
    if not isinstance(time_values, list): time_values = [time_values]

    if broken_axes:
        fig = plt.figure(figsize=(8, 6))
        ax = brokenaxes(xlims=broken_xlims, wspace=broken_width)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if color:
        colors = color

    num_time_vals = len(time_values)

    # --- 2. Iterate through datasets and time values ---
    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        base_color = colors[i % len(colors)]
        
        try:
            irf_width_offset = ds['irf_width'].item()
        except KeyError:
            print(f"Warning: 'irf_width' not found in '{ds_label}'. Assuming width offset is 0.")
            irf_width_offset = 0
        
        if measurement_type == "TRPL":
            if normalize:
                max_val = np.zeros(num_time_vals,)                  
                for j, relative_time in enumerate(time_values):
                    try:
                        irf_offset = ds['irf_center'].item()
                        # UPDATED: Added - irf_width_offset
                        absolute_time_to_select = relative_time + irf_offset - irf_width_offset
                        fitted_slice = ds['fitted_data'].sel(time=absolute_time_to_select, method='nearest').squeeze()
                        max_val[j] = fitted_slice[fitted_slice.argmax()]
                    except KeyError:
                        print(f"Warning: 'irf_center' not found in '{ds_label}' for TRPL mode. Assuming offset is 0.")
                        absolute_time_to_select = relative_time - irf_width_offset
                        fitted_slice = ds['fitted_data'].sel(time=absolute_time_to_select, method='nearest').squeeze()
                        max_val[j] = fitted_slice[fitted_slice.argmax()]     
                norm_val = np.max(max_val)
            
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
                    if normalize:
                        # Find the value with the maximum absolute magnitude from the fit
                        fitted_slice = fitted_slice / norm_val
                        data_slice = data_slice / norm_val
                        
                # --- 4. Plotting logic (common for both types) ---
                lightness_factor = 1.0
                if num_time_vals > 1:
                    lightness_factor = 0.7 + (j / (num_time_vals - 1)) * 0.6
                h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                plot_color = colorsys.hls_to_rgb(h, max(0, min(1, l * lightness_factor)), s)

                legend_label = f"{ds_label} {relative_time:.1f} ps"
                if plot_raw:
                    lines = ax.plot(ds['spectral'], data_slice, label=legend_label, color=plot_color, linewidth=2)                                 
                else:
                    if broken_axes:
                        # 1. Do not use 'line, =' unpacking
                        lines = ax.plot(ds['spectral'], fitted_slice, label=legend_label, color=plot_color, linewidth=2)
                        
                        # 2. Extract color from the first line of the first broken axis segment
                        line_color = lines[0][0].get_color()
                        
                        # 3. Increase zorder so it sits above the background
                        ax.scatter(ds['spectral'], data_slice, color=line_color, alpha=0.5, s=10, zorder=2)
                    else:
                        line, = ax.plot(ds['spectral'], fitted_slice, label=legend_label, color=plot_color, linewidth=2)
                        ax.scatter(ds['spectral'], data_slice, color=line.get_color(), alpha=0.5, s=10, zorder=-1)
                if export:
                    path = Path(export_folder)
                    path.mkdir(parents=True, exist_ok=True)
                    export_var = data_slice.to_dataframe()
                    export_var.to_csv(export_folder+"/"+legend_label+"spectrum.csv")
                    

            except Exception as e:
                print(f"Could not plot for {ds_label} at time {relative_time}: {e}")

    # --- 5. Final plot formatting ---
    title = f"Spectral Slices ({measurement_type} Mode)"
    legend_title = "Dataset (Time)"
    if measurement_type == "TA":
        legend_title = "Dataset (Time relative to t₀)"
        ax.set_ylabel("ΔA (mOD)")
    elif measurement_type == "TRPL":
        legend_title = "Dataset (Time relative to IRF)"
        if normalize:
            ax.set_ylabel("Normalized Intensity (A.U.)")
        else:
            ax.set_ylabel("I (A.U.)")
            
    #ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    if legend:
        ax.legend(frameon=False)
    #ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    if xlim:
        if broken_axes:
            return
        else:
            ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if hide_spines:
        # Hide specific spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    #format_publication_plot_no_latex(ax=ax)
    if return_fig_object:
        return ax
    else:
        plt.tight_layout()
        plt.show()