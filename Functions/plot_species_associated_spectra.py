import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

def plot_species_associated_spectra(datasets, dataset_labels, measurement_type="TA", normalize=False, species_to_plot=None, xlim=None, ylim=None):
    """
    Plots selected species-associated spectra from one or more datasets.

    Each dataset has a unique base color. Species are differentiated by varying
    the lightness of the base color.

    Args:
        datasets (xr.Dataset or list): A single dataset or a list of datasets.
        dataset_labels (str or list): Labels for the datasets.
        species_to_plot (list, optional): A list of species names (strings) to plot.
                                          If None, all species are plotted. Defaults to None.
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
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # --- 3. Iterate through each dataset and its species ---
    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        try:
            sas = ds['species_associated_spectra']
            spectral_coords = ds['spectral']
            
            if species_to_plot:
                # Filter the available species based on the user's list
                available_species = sas.coords['species'].values
                plot_list = [s for s in species_to_plot if s in available_species]
                if not plot_list:
                    print(f"Warning: None of the requested species found in '{ds_label}'. Skipping.")
                    continue
            else:
                # Default to plotting all available species
                plot_list = sas.coords['species'].values
            
            num_species_to_plot = len(plot_list)
            base_color = colors[i % len(colors)]
            
            # Iterate through the determined list of species
            for j, species_name in enumerate(plot_list):
                # Modify color lightness for each species
                lightness_factor = 1.0
                if num_species_to_plot > 1:
                    lightness_factor = 0.7 + (j / (num_species_to_plot - 1)) * 0.6
                                                          
                h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                plot_color = colorsys.hls_to_rgb(h, max(0, min(1, l * lightness_factor)), s)
                
                spectrum_slice = sas.sel(species=species_name)
                
                if normalize:
                    if num_species_to_plot > 1:
                        norm_arr = np.zeros(len(plot_list))
                        for j, species_name in enumerate(plot_list):
                            np_fitted = sas.sel(species=species_name).values
                            norm_arr[j] = np_fitted[np.abs(np_fitted).argmax()]
                        norm_val = norm_arr.max()
                        if norm_val != 0: # Avoid division by zero
                                spectrum_slice = spectrum_slice / norm_val
                        
                    else:                        
                        # Find the value with the maximum absolute magnitude from the fit
                        np_fitted = spectrum_slice.values
                        if np_fitted.size > 0:
                            norm_val = np_fitted[np.abs(np_fitted).argmax()]
                            if norm_val != 0: # Avoid division by zero
                                spectrum_slice = spectrum_slice / norm_val
                            
                legend_label = f"{ds_label} ({species_name})"
                
                ax.plot(spectral_coords, spectrum_slice, label=legend_label, 
                        color=plot_color, linewidth=2.5)

        except KeyError:
            print(f"Warning: 'species_associated_spectra' not found in '{ds_label}'. Skipping.")
    
    # --- 4. Final plot formatting ---
    ax.set_title("Species-Associated Spectra")
    ax.set_xlabel("Wavelength (nm)")
    if measurement_type == "TA":
        ax.set_ylabel("ΔA (mOD)")
    else:
        ax.set_ylabel("I (A.U.)")

    ax.legend(title="Dataset (Species)")
    #ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    
    # --- 5. Set axis limits if provided ---
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    #format_publication_plot_no_latex(ax=ax)
    plt.show()