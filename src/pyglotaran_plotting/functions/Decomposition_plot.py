import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cycler import cycler
# from pyfonts import load_google_font
from pyglotaran_extras.plotting.utils import (
    MinorSymLogLocator,
    extract_irf_location,
    shift_time_axis_by_irf_location,
)


def plot_decomposition(result, dataname, wavelength):
    """
    Processes and plots the decomposition of transient absorption data for a specific wavelength.

    This function categorizes model components, constructs a dataset with these components,
    and generates a plot showing the data, the fit, and the contribution of each component
    (e.g., Damped Oscillation, Coherent Artifact) at the specified wavelength.

    Parameters
    ----------
    result : pyglotaran.Result
        The result object from a pyglotaran analysis.
    dataname : str
        The name of the dataset within the result object to be plotted.
    wavelength : int or float
        The specific wavelength to plot the kinetic traces for.
    """
    # 1. Extract data and CLPs from the result object
    matrix = result.data[dataname].matrix
    clp = result.data[dataname].clp

    # 2. Categorize CLP labels into different components
    ca_labels = []
    ca_doas_labels = []
    doas_labels = []
    non_doas_labels = []

    for clp_label in matrix.clp_label:
        label_str = clp_label.item()
        if label_str.startswith("coherent_artifact_"):
            ca_labels.append(label_str)
        elif label_str.startswith(("osc5_", "osc6_")):
            ca_doas_labels.append(label_str)
        elif label_str.endswith(("_sin", "_cos")):
            doas_labels.append(label_str)
        else:
            non_doas_labels.append(label_str)

    # 3. Build a dictionary of xarray.DataArray for plotting
    data_dict = {
        "Data": result.data[dataname].data,
        "Fitted data": result.data[dataname].fitted_data,
    }

    # Add individual non-DOAS components
    for non_doas_label in non_doas_labels:
        data_dict[non_doas_label] = (
            matrix.sel(clp_label=non_doas_label) * clp.sel(clp_label=non_doas_label)
        ).drop_vars("clp_label")

    # Add summed components
    data_dict["Without Doas"] = (
        (matrix.sel(clp_label=non_doas_labels)) * clp.sel(clp_label=non_doas_labels)
    ).sum(dim="clp_label")
    
    data_dict["Damped Oscillation"] = (
        (matrix.sel(clp_label=doas_labels) - 1) * clp.sel(clp_label=doas_labels)
    ).sum(dim="clp_label")
    
    data_dict["Coherent Artifact"] = (
        matrix.sel(clp_label=ca_labels) * clp.sel(clp_label=ca_labels)
    ).sum(dim="clp_label") + (
        (matrix.sel(clp_label=ca_doas_labels) - 1) * clp.sel(clp_label=ca_doas_labels)
    ).sum(dim="clp_label")

    data = xr.Dataset(data_dict)

    # # 4. Configure plot aesthetics (fonts and styles)
    # def configure_matplotlib_fonts(font_name='Open Sans'):
    #     load_google_font(font_name)
    #     plt.rcParams.update({
    #         'font.family': 'sans-serif',
    #         'font.sans-serif': [font_name],
    #         'font.size': 18,
    #         'text.color': '#1F2937',
    #         'axes.labelcolor': '#1F2937',
    #         'axes.edgecolor': '#1F2937',
    #         'xtick.major.size': 4.5,
    #         'ytick.major.size': 4.5,
    #     })

    # configure_matplotlib_fonts()

    # 5. Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    myFRLcolors = [
        cmc.batlowS(1, alpha=0.8), "black", "k", "midnightblue", "blue",
        "dodgerblue", "darkred", "olive", "aquamarine", "y", "tab:brown", "tab:purple"
    ]
    ax.set_prop_cycle(cycler(color=myFRLcolors))

    # Plot each component
    for key, data_array in data.data_vars.items():
        # Select the data at the specified wavelength
        trace = data_array.sel(spectral=wavelength, method="nearest")
        # Shift time axis by the Instrument Response Function (IRF) location
        irf_location = extract_irf_location(result.data[dataname], wavelength)
        shifted_trace = shift_time_axis_by_irf_location(trace, irf_location=irf_location)
        shifted_trace.plot(x="time", ax=ax, label=key)

    # 6. Customize the plot axes and labels
    ax.set_xscale("symlog", linthresh=10)
    ax.xaxis.set_minor_locator(MinorSymLogLocator(1e-9)) # Adjusted for symlog
    
    ax.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        mode="expand", borderaxespad=0, ncol=4, frameon=False
    )
    
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("ΔA (mOD)")
    ax.set_title(f"Kinetic Trace at {wavelength} nm")
    ax.set_xlim(-2, 100)
    
    for vline_pos in [0, 10]:
        ax.axvline(vline_pos, color="k", linewidth=1, linestyle='--')

    fig.tight_layout()

    # 7. Save the figure with a dynamic filename
    output_filename = f'decomposition_{wavelength}_nm_transparent.png'
    plt.savefig(output_filename, format='png', dpi=600, transparent=True)
    print(f"Plot saved as {output_filename}")
    plt.show()


# --- Example Usage ---
# Assuming you have a 'result' object and 'dataname' defined from a pyglotaran analysis.
# For example:
# from glotaran.io import load_result
# result = load_result("path/to/your/result.yml")
# dataname = "your_dataset_name"

# Then you can call the function for any wavelength:
# plot_decomposition(result, dataname, wavelength=450)
# plot_decomposition(result, dataname, wavelength=550)