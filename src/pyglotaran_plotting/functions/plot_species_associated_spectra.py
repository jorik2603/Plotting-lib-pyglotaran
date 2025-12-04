import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

def plot_species_associated_spectra(datasets, dataset_labels, measurement_type="TA", 
                                    normalize=False, species_to_plot=None, xlim=None, ylim=None):
    """
    Plots species-associated spectra (SAS) from one or more datasets.

    Differentiates datasets by base color and species within a dataset by lightness.

    Parameters
    ----------
    datasets : list or xarray.Dataset
        The dataset(s) containing 'species_associated_spectra'.
    dataset_labels : list or str
        Labels for the datasets.
    measurement_type : str, optional
        "TA" (Transient Absorption) or "TRPL". Used for axis labeling. Default is "TA".
    normalize : bool, optional
        If True, normalizes each spectrum to its max value. Default is False.
    species_to_plot : list of str, optional
        Specific species names to plot. If None, plots all available species.
    xlim, ylim : tuple, optional
        Limits for x (wavelength) and y (amplitude) axes.

    Returns
    -------
    None
        Displays the plot.
    """
    # Standardize inputs
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(dataset_labels, list):
        dataset_labels = [dataset_labels]

    if len(datasets) != len(dataset_labels):
        raise ValueError("The number of datasets must match the number of labels.")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        try:
            sas = ds['species_associated_spectra']
            spectral_coords = ds['spectral']
            
            # Filter species
            if species_to_plot:
                available_species = sas.coords['species'].values
                plot_list = [s for s in species_to_plot if s in available_species]
                if not plot_list:
                    print(f"Warning: No requested species found in '{ds_label}'.")
                    continue
            else:
                plot_list = sas.coords['species'].values
            
            num_species_to_plot = len(plot_list)
            base_color = colors[i % len(colors)]
            
            for j, species_name in enumerate(plot_list):
                # Calculate color lightness variation
                lightness_factor = 1.0
                if num_species_to_plot > 1:
                    lightness_factor = 0.7 + (j / (num_species_to_plot - 1)) * 0.6
                                                          
                h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                plot_color = colorsys.hls_to_rgb(h, max(0, min(1, l * lightness_factor)), s)
                
                spectrum_slice = sas.sel(species=species_name)
                
                # Normalization Logic
                if normalize:
                    # Normalize based on the max of ALL plotted species in this dataset, or individually?
                    # The original logic normalized individually if single species, 
                    # or by the max of all selected species if multiple.
                    if num_species_to_plot > 1:
                        norm_arr = np.zeros(len(plot_list))
                        for k, s_name in enumerate(plot_list):
                            val = sas.sel(species=s_name).values
                            norm_arr[k] = val[np.abs(val).argmax()]
                        norm_val = norm_arr.max()
                        if norm_val != 0:
                            spectrum_slice = spectrum_slice / norm_val
                    else:                        
                        np_fitted = spectrum_slice.values
                        if np_fitted.size > 0:
                            norm_val = np_fitted[np.abs(np_fitted).argmax()]
                            if norm_val != 0:
                                spectrum_slice = spectrum_slice / norm_val
                            
                legend_label = f"{ds_label} ({species_name})"
                
                ax.plot(spectral_coords, spectrum_slice, label=legend_label, 
                        color=plot_color, linewidth=2.5)

        except KeyError:
            print(f"Warning: 'species_associated_spectra' not found in '{ds_label}'. Skipping.")
    
    ax.set_title("Species-Associated Spectra")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("ΔA (mOD)" if measurement_type == "TA" else "I (A.U.)")

    ax.legend(title="Dataset (Species)")
    ax.axhline(0, color='black', linewidth=0.5)
    
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.show()