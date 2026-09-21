from pathlib import Path
import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import savgol_filter
import xarray as xr


def plot_multi_time_traces(
    datasets,
    dataset_labels,
    spectral_values,
    measurement_type="TA",
    normalize=False,
    normalize_raw=False,
    normalize_per_dataset=False,
    rescale=False,
    apply_chirp_correction=False,
    xlim=None,
    ylim=None,
    smoothing=False,
    sg_window=5,
    sg_order=0,
    symlog_time=False,
    log_y=False,
    linthresh=1,
    color=None,
    export=False,
    export_folder="time_traces",
    return_fig_object=False,
    hide_spines=False,
    legend=False,
    simple_legend=False,
    single_legend_entry=False,
    raw_as_line=False,
    
):
  """Plots time traces with specific time-zero logic for TA or TRPL measurements.

  Args:
      datasets (list): A list of datasets.
      dataset_labels (list): A list of labels for the datasets.
      spectral_values (list): Spectral values to plot.
      measurement_type (str): "TA" or "TRPL". Determines how time-zero is
        defined.
      normalize (bool): If True plots normalized data.
      normalize_raw (bool): If True uses raw data values for normalization.
      normalize_per_dataset (bool): If True, normalizes all traces within a
        dataset by the absolute maximum value found across all requested
        spectral traces for that dataset.
      rescale (bool): If True will rescale normalization based on min/max values
        of the fit.
      apply_chirp_correction (bool): If True and type is "TA", applies a
        spectrally-dependent time shift.
      xlim (tuple, optional): A tuple (min, max) for the x-axis limits.
      ylim (tuple, optional): A tuple (min, max) for the y-axis limits.
      raw_as_line (bool, optional): If True, plots raw data as a low-alpha line
        instead of a scatter plot.
  """
  if measurement_type not in ["TA", "TRPL"]:
    raise ValueError("measurement_type must be either 'TA' or 'TRPL'.")
  if not isinstance(datasets, list):
    datasets = [datasets]
  if not isinstance(dataset_labels, list):
    dataset_labels = [dataset_labels]
  if not isinstance(spectral_values, list):
    spectral_values = [spectral_values]

  fig, ax = plt.subplots(figsize=(8, 6))

  colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  if color:
    colors = color

  # Initialize a global color counter so every trace gets a unique color
  color_index = 0

  for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):

    # Get offsets for this dataset, with fallbacks
    try:
      irf_width_offset = ds["irf_width"].item()
    except KeyError:
      irf_width_offset = 0

    try:
      irf_center_offset = ds["irf_center"].item()
    except KeyError:
      irf_center_offset = 0

    # Calculate dataset-level absolute max if requested
    dataset_norm_val = 1.0
    if normalize and normalize_per_dataset:
      ds_max_abs = 0
      for spec_val in spectral_values:
        try:
          if normalize_raw:
            vals = (
                ds["data"].sel(spectral=spec_val, method="nearest").values
                * 1000
            )
          else:
            vals = (
                ds["fitted_data"]
                .sel(spectral=spec_val, method="nearest")
                .values
                * 1000
            )
          if vals.size > 0:
            local_max = np.max(np.abs(vals))
            if local_max > ds_max_abs:
              ds_max_abs = local_max
        except Exception:
          continue
      dataset_norm_val = ds_max_abs if ds_max_abs != 0 else 1.0

    for j, spec_val in enumerate(spectral_values):
      try:
        # --- Time axis calculation logic ---
        time_coords_base = ds["time"].copy()

        # --- TA Logic ---
        if measurement_type == "TA":
          if apply_chirp_correction:
            try:
              chirp_offset = (
                  ds["irf_center_location"]
                  .sel(spectral=spec_val, method="nearest")
                  .item()
              )
              time_coords_for_plot = (
                  time_coords_base - chirp_offset + irf_width_offset
              )
            except KeyError:
              print(
                  f"Warning: 'irf_center_location' not found in '{ds_label}'."
                  " Cannot apply chirp correction."
              )
              continue
          else:
            time_coords_for_plot = time_coords_base

        # --- TRPL Logic ---
        elif measurement_type == "TRPL":
          time_coords_for_plot = (
              time_coords_base - irf_center_offset + irf_width_offset
          )

        # Assign a completely distinct color to each plotted trace
        plot_color = colors[color_index % len(colors)]
        color_index += 1

        # Select data and plot
        data_slice = ds["data"].sel(spectral=spec_val, method="nearest") * 1000
        fitted_slice = (
            ds["fitted_data"].sel(spectral=spec_val, method="nearest") * 1000
        )
        actual_spec_val = fitted_slice["spectral"].item()

        if single_legend_entry:
          if j == 0:
            if simple_legend:
              legend_label = f"{ds_label}"
            else:
              legend_label = f"{ds_label} {actual_spec_val:.0f} nm"
          else:
            legend_label = "_nolegend_"
        else:
          if simple_legend:
            legend_label = f"{ds_label}"
          else:
            legend_label = f"{ds_label} {actual_spec_val:.0f} nm"

        if normalize:
          np_fitted = fitted_slice.values
          np_data = data_slice.values
          if np_fitted.size > 0:
            if normalize_per_dataset:
              norm_val = dataset_norm_val
            else:
              target_array = np_data if normalize_raw else np_fitted
              norm_val = target_array[np.abs(target_array).argmax()]
              if np.abs(target_array).argmax() != target_array.argmax():
                norm_val = -1 * abs(norm_val)

            if norm_val != 0:
              if rescale:
                min_val = np_fitted[np.abs(np_fitted).argmin()]
                data_slice = (data_slice - min_val) / (norm_val - min_val)
                fitted_slice = (fitted_slice - min_val) / (norm_val - min_val)
              else:
                data_slice = data_slice / norm_val
                fitted_slice = fitted_slice / norm_val

        if smoothing:
          fitted_slice = savgol_filter(
              data_slice, window_length=sg_window, polyorder=sg_order
          )

        line, = ax.plot(
            time_coords_for_plot,
            fitted_slice,
            label=legend_label,
            color=plot_color,
            linewidth=2,
        )

        if raw_as_line:
          ax.plot(
              time_coords_for_plot,
              data_slice,
              color=line.get_color(),
              alpha=0.5,
              zorder=-1,
          )
        else:
          ax.scatter(
              time_coords_for_plot,
              data_slice,
              color=line.get_color(),
              alpha=0.5,
              s=10,
              zorder=-1,
          )

        if export:
          path = Path(export_folder)
          path.mkdir(parents=True, exist_ok=True)
          export_data = data_slice.values
          export_fit = fitted_slice.values
          export_time = time_coords_for_plot.values
          export_var = np.column_stack((export_time, export_data, export_fit))
          np.savetxt(
              export_folder + "/" + legend_label + "extracted.csv",
              export_var,
              delimiter=",",
          )

      except Exception as e:
        print(f"Could not plot for {ds_label} at {spec_val}: {e}")

  # Final plot formatting
  xlabel = "Time / ps"
  if measurement_type == "TA":
    if normalize:
      ax.set_ylabel("Norm. ΔAbs / a.u.")
    else:
      ax.set_ylabel("ΔAbs / mOD")
  elif measurement_type == "TRPL":
    if normalize:
      ax.set_ylabel("Normalized Intensity (A.U.)")
    else:
      ax.set_ylabel("I (A.U.)")

  ax.set_xlabel(xlabel)
  if legend:
    ax.legend(frameon=False)
  if symlog_time:
    ax.set_xscale("symlog", linthresh=linthresh)
  if log_y:
    ax.set_yscale("log")
  ax.axhline(0, color="black", linewidth=0.5)
  if xlim:
    ax.set_xlim(xlim)
  if ylim:
    ax.set_ylim(ylim)
  if hide_spines:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

  if return_fig_object:
    return ax
  else:
    plt.tight_layout()
    plt.show()