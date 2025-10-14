def plot_species_associated_spectra(datasets, dataset_labels, xlim=None, ylim=None):
    """
    Plots species-associated spectra from one or more datasets.

    Each dataset is assigned a unique color. Different species within the
    same dataset are differentiated by unique line styles.

    Args:
        datasets (xr.Dataset or list): A single dataset or a list of datasets.
                                     Each should contain 'species_associated_spectra'.
        dataset_labels (str or list): A label or list of labels for the datasets.
        xlim (tuple, optional): A tuple (min, max) to set the x-axis limits.
        ylim (tuple, optional): A tuple (min, max) to set the y-axis limits.
    """
    # --- 1. Standardize inputs and define plot styles ---
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(dataset_labels, list):
        dataset_labels = [dataset_labels]

    if len(datasets) != len(dataset_labels):
        raise ValueError("The number of datasets must match the number of labels.")

    # Define a cycle of linestyles for the species
    linestyles = ['-', '--', ':', '-.']
    # Get the default color cycle from matplotlib
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- 2. Create the plot ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # --- 3. Iterate through each dataset and its species ---
    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        try:
            sas = ds['species_associated_spectra']
            spectral_coords = ds['spectral']
            
            # Assign a unique color to the current dataset
            color = colors[i % len(colors)]
            
            # Iterate through each species, assigning a unique linestyle
            for j, species_name in enumerate(sas.coords['species'].values):
                spectrum_slice = sas.sel(species=species_name)
                linestyle = linestyles[j % len(linestyles)]
                
                legend_label = f"{ds_label} ({species_name})"
                
                # Plot using the assigned color and linestyle
                ax.plot(spectral_coords, spectrum_slice, label=legend_label, 
                        color=color, linestyle=linestyle, linewidth=2.5)

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