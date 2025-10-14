import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

def plot_species_associated_spectra(datasets, dataset_labels, xlim=None, ylim=None):
    """
    Plots species-associated spectra from one or more datasets.

    Each dataset is assigned a unique base color. Different species within the
    same dataset are differentiated by slightly varying the lightness of the base color.

    Args:
        datasets (xr.Dataset or list): A single dataset or a list of datasets.
                                     Each should contain 'species_associated_spectra'.
        dataset_labels (str or list): A label or list of labels for the datasets.
        xlim (tuple, optional): A tuple (min, max) to set the x-axis limits.
        ylim (tuple, optional): A tuple (min, max) to set the y-axis limits.
    """
    # --- 1. Standardize inputs and get color cycle ---
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(dataset_labels, list):
        dataset_labels = [dataset_labels]

    if len(datasets) != len(dataset_labels):
        raise ValueError("The number of datasets must match the number of labels.")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- 2. Create the plot ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # --- 3. Iterate through each dataset and its species ---
    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        try:
            sas = ds['species_associated_spectra']
            spectral_coords = ds['spectral']
            num_species = len(sas.coords['species'])
            
            # Assign a unique base color to the current dataset
            base_color = colors[i % len(colors)]
            
            # Iterate through each species, assigning a variation of the base color
            for j, species_name in enumerate(sas.coords['species'].values):
                # Modify color lightness for each species
                lightness_factor = 1.0
                if num_species > 1:
                    # Map index j to a lightness multiplier (e.g., from 0.7 to 1.3)
                    lightness_factor = 0.7 + (j / (num_species - 1)) * 0.6
                
                h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                plot_color = colorsys.hls_to_rgb(h, max(0, min(1, l * lightness_factor)), s)
                
                spectrum_slice = sas.sel(species=species_name)
                legend_label = f"{ds_label} ({species_name})"
                
                # Plot using the calculated color
                ax.plot(spectral_coords, spectrum_slice, label=legend_label, 
                        color=plot_color, linewidth=2.5)

        except KeyError:
            print(f"Warning: 'species_associated_spectra' not found in '{ds_label}'. Skipping.")

    # --- 4. Final plot formatting ---
    ax.set_title("Species-Associated Spectra")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend(title="Dataset (Species)", fontsize='medium')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    
    # --- 5. Set axis limits if provided ---
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.show()