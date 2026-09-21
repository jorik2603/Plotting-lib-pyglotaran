import colorsys
from pathlib import Path
from brokenaxes import brokenaxes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_multi_spectral_slices(
    datasets,
    dataset_labels,
    time_values,
    measurement_type="TA",
    plot_raw=False,
    apply_chirp_correction=False,
    legend=True,
    color=None,
    normalize=False,
    xlim=None,
    ylim=None,
    broken_axes=False,
    broken_xlims=None,
    broken_width=0.1,
    export=False,
    export_folder="slices",
    return_fig_object=False,
    hide_spines=False,
    simple_legend=False,
    explicit_legend=None
):
    """Plots spectral slices with specific logic for TA or TRPL measurements.

    Args:
        datasets (list): A list of datasets.
        dataset_labels (list): A list of labels for the datasets.
        time_values (list): Time values to plot. Interpretation depends on
          measurement_type.
        measurement_type (str): "TA" or "TRPL". Determines how time-zero is
          defined.
        apply_chirp_correction (bool): If True and type is "TA", applies a
          spectrally-dependent time shift.
        normalize (bool): If True normalizes spectrum now only for TRPL
          measurement type.
        color (str, list, tuple, np.ndarray, optional): Explicit color(s). Can
          be a single color, a list matching len(time_values), a list matching
          len(datasets), a flat list of all traces (len(datasets)*len(times)),
          or a 2D list of shape (len(datasets), len(times)).
        xlim (tuple, optional): A tuple (min, max) for the x-axis limits.
        ylim (tuple, optional): A tuple (min, max) for the y-axis limits.
        broken_axes (bool): If True plots broken axes
        broken_xlims ((tuple,tuple),optional): Two tuples (min, max) for the
          x-axis limits when using brokenaxes.
    """
    # --- 1. Validate inputs and set up plot ---
    if measurement_type not in ["TA", "TRPL"]:
        raise ValueError("measurement_type must be either 'TA' or 'TRPL'.")
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(dataset_labels, list):
        dataset_labels = [dataset_labels]
    if not isinstance(time_values, list):
        time_values = [time_values]

    if broken_axes:
        fig = plt.figure(figsize=(8, 6))
        ax = brokenaxes(xlims=broken_xlims, wspace=broken_width)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    num_datasets = len(datasets)
    num_time_vals = len(time_values)

    # --- Helper to detect if a variable represents a single color ---
    def _is_single_color(c):
        if isinstance(c, str):
            return True
        if isinstance(c, np.ndarray):
            return c.ndim == 1 and len(c) in (3, 4)
        if isinstance(c, (tuple, list)) and len(c) in (3, 4):
            return all(isinstance(x, (int, float, np.number)) for x in c)
        return False

    # --- Determine Color Mapping ---
    explicit_trace_colors = False
    custom_colors = None
    base_colors = None

    if color is not None:
        if _is_single_color(color):
            # Single color: apply lightness gradient across time points
            base_colors = [color] * num_datasets
        elif isinstance(color, (list, tuple, np.ndarray)):
            # Case A: Nested list/2D array [dataset_idx][time_idx]
            if (
                len(color) == num_datasets
                and isinstance(color[0], (list, tuple, np.ndarray))
                and len(color[0]) == num_time_vals
                and not _is_single_color(color[0])
            ):
                explicit_trace_colors = True
                custom_colors = color

            # Case B: Flat list matching total traces across all datasets and time values
            elif len(color) == (num_datasets * num_time_vals) and (
                num_datasets > 1 or num_time_vals > 1
            ):
                explicit_trace_colors = True
                custom_colors = [
                    color[i * num_time_vals : (i + 1) * num_time_vals]
                    for i in range(num_datasets)
                ]

            # Case C: 1D list matching the number of time values (e.g. colormap for times)
            elif len(color) == num_time_vals:
                explicit_trace_colors = True
                custom_colors = [list(color) for _ in range(num_datasets)]

            # Case D: 1D list matching the number of datasets
            elif len(color) == num_datasets:
                base_colors = color

            # Fallback: Cycle provided colors across all traces
            else:
                explicit_trace_colors = True
                flat = [
                    color[k % len(color)]
                    for k in range(num_datasets * num_time_vals)
                ]
                custom_colors = [
                    flat[i * num_time_vals : (i + 1) * num_time_vals]
                    for i in range(num_datasets)
                ]
    else:
        # Default: Cycle matplotlib palette per dataset with lightness gradient per time
        prop_cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        base_colors = [
            prop_cycle_colors[i % len(prop_cycle_colors)]
            for i in range(num_datasets)
        ]

    # --- 2. Iterate through datasets and time values ---
    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):

        try:
            irf_width_offset = ds["irf_width"].item()
        except KeyError:
            print(
                f"Warning: 'irf_width' not found in '{ds_label}'. Assuming"
                " width offset is 0."
            )
            irf_width_offset = 0

        if measurement_type == "TRPL":
            if normalize:
                max_val = np.zeros(
                    num_time_vals,
                )
                for j, relative_time in enumerate(time_values):
                    try:
                        irf_offset = ds["irf_center"].item()
                        absolute_time_to_select = (
                            relative_time + irf_offset - irf_width_offset
                        )
                        fitted_slice = (
                            ds["fitted_data"]
                            .sel(
                                time=absolute_time_to_select, method="nearest"
                            )
                            .squeeze()
                        )
                        max_val[j] = fitted_slice[fitted_slice.argmax()]
                    except KeyError:
                        print(
                            f"Warning: 'irf_center' not found in '{ds_label}'"
                            " for TRPL mode. Assuming offset is 0."
                        )
                        absolute_time_to_select = (
                            relative_time - irf_width_offset
                        )
                        fitted_slice = (
                            ds["fitted_data"]
                            .sel(
                                time=absolute_time_to_select, method="nearest"
                            )
                            .squeeze()
                        )
                        max_val[j] = fitted_slice[fitted_slice.argmax()]
                norm_val = np.max(max_val)

        for j, relative_time in enumerate(time_values):
            # --- 3. Determine selection time based on measurement_type ---
            try:
                # --- TA Logic ---
                if measurement_type == "TA":
                    if apply_chirp_correction:
                        try:
                            chirp_data = ds["irf_center_location"]
                            absolute_times_to_select = (
                                relative_time + chirp_data - irf_width_offset
                            )
                            data_slice = (
                                ds["data"]
                                .sel(
                                    time=absolute_times_to_select,
                                    method="nearest",
                                )
                                .squeeze()
                            )
                            fitted_slice = (
                                ds["fitted_data"]
                                .sel(
                                    time=absolute_times_to_select,
                                    method="nearest",
                                )
                                .squeeze()
                            )
                        except KeyError:
                            print(
                                "Warning: 'irf_center_location' not found in"
                                f" '{ds_label}'. Cannot apply chirp correction."
                            )
                            continue
                    else:
                        absolute_time_to_select = relative_time
                        data_slice = (
                            ds["data"]
                            .sel(
                                time=absolute_time_to_select, method="nearest"
                            )
                            .squeeze()
                        )
                        fitted_slice = (
                            ds["fitted_data"]
                            .sel(
                                time=absolute_time_to_select, method="nearest"
                            )
                            .squeeze()
                        )

                # --- TRPL Logic ---
                elif measurement_type == "TRPL":
                    try:
                        irf_offset = ds["irf_center"].item()
                        absolute_time_to_select = (
                            relative_time + irf_offset - irf_width_offset
                        )
                        data_slice = (
                            ds["data"]
                            .sel(
                                time=absolute_time_to_select, method="nearest"
                            )
                            .squeeze()
                        )
                        fitted_slice = (
                            ds["fitted_data"]
                            .sel(
                                time=absolute_time_to_select, method="nearest"
                            )
                            .squeeze()
                        )
                    except KeyError:
                        print(
                            f"Warning: 'irf_center' not found in '{ds_label}'"
                            " for TRPL mode. Assuming offset is 0."
                        )
                        absolute_time_to_select = (
                            relative_time - irf_width_offset
                        )
                        data_slice = (
                            ds["data"]
                            .sel(
                                time=absolute_time_to_select, method="nearest"
                            )
                            .squeeze()
                        )
                        fitted_slice = (
                            ds["fitted_data"]
                            .sel(
                                time=absolute_time_to_select, method="nearest"
                            )
                            .squeeze()
                        )
                    if normalize:
                        fitted_slice = fitted_slice / norm_val
                        data_slice = data_slice / norm_val

                # --- 4. Resolve Color for Trace ---
                if explicit_trace_colors:
                    plot_color = custom_colors[i][j]
                else:
                    base_color = base_colors[i]
                    lightness_factor = 1.0
                    if num_time_vals > 1:
                        lightness_factor = (
                            0.7 + (j / (num_time_vals - 1)) * 0.6
                        )
                    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                    plot_color = colorsys.hls_to_rgb(
                        h, max(0, min(1, l * lightness_factor)), s
                    )

                if simple_legend:
                    legend_label = f"{ds_label}"
                else:
                    legend_label = f"{ds_label} {relative_time:.0f} ps"
                    if explicit_legend:
                        legend_label = explicit_legend[j]

                if plot_raw:
                    lines = ax.plot(
                        ds["spectral"],
                        data_slice,
                        label=legend_label,
                        color=plot_color,
                        linewidth=2,
                    )
                else:
                    if broken_axes:
                        lines = ax.plot(
                            ds["spectral"],
                            fitted_slice,
                            label=legend_label,
                            color=plot_color,
                            linewidth=2,
                        )
                        line_color = lines[0][0].get_color()
                        ax.scatter(
                            ds["spectral"],
                            data_slice,
                            color=line_color,
                            alpha=0.5,
                            s=10,
                            zorder=2,
                        )
                    else:
                        (line,) = ax.plot(
                            ds["spectral"],
                            fitted_slice,
                            label=legend_label,
                            color=plot_color,
                            linewidth=2,
                        )
                        ax.scatter(
                            ds["spectral"],
                            data_slice,
                            color=line.get_color(),
                            alpha=0.5,
                            s=10,
                            zorder=-1,
                        )

                if export:
                    path = Path(export_folder)
                    path.mkdir(parents=True, exist_ok=True)
                    export_var = data_slice.to_dataframe()
                    export_fit = fitted_slice.to_dataframe()
                    export_var.to_csv(
                        export_folder + "/" + legend_label + "raw_spectrum.csv"
                    )
                    export_fit.to_csv(
                        export_folder + "/" + legend_label + "fit_spectrum.csv"
                    )

            except Exception as e:
                print(
                    f"Could not plot for {ds_label} at time {relative_time}:"
                    f" {e}"
                )

    # --- 5. Final plot formatting ---
    if measurement_type == "TA":
        ax.set_ylabel("ΔAbs / mOD")
    elif measurement_type == "TRPL":
        if normalize:
            ax.set_ylabel("Normalized Intensity (A.U.)")
        else:
            ax.set_ylabel("I (A.U.)")

    ax.set_xlabel("Wavelength / nm")
    if legend:
        ax.legend(frameon=False)
    ax.axhline(0, color="black", linewidth=0.5)

    if xlim:
        if not broken_axes:
            ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if hide_spines and not broken_axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if return_fig_object:
        return ax
    else:
        plt.tight_layout()
        plt.show()