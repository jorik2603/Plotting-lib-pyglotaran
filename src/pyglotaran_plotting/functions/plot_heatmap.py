import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import interpolate

# Attempt to import the preferred colormap library
try:
    import cmcrameri.cm as cmc
    DEFAULT_CMAP = cmc.vik
except ImportError:
    print("Warning: 'cmcrameri' library not found. Falling back to 'RdBu_r' colormap. Run 'pip install cmcrameri' for perceptually uniform colormaps.")
    DEFAULT_CMAP = 'RdBu_r'

def _apply_chirp_correction_to_data(data_array, time_coords, spectral_coords, 
                                    irf_center_location_array,irf_width):
    """
    Applies chirp correction to the 2D data array using 1D interpolation
    along the time axis for each wavelength.
    """
    t = time_coords
    wv = spectral_coords
    d = data_array.values  # Shape (time, spectral)
    d_corr = np.zeros_like(d)

    if len(wv) != len(irf_center_location_array):
        raise ValueError(
            f"Chirp data length ({len(irf_center_location_array)}) does not match "
            f"spectral coordinate length ({len(wv)})."
        )

    for i in range(len(wv)):
        correcttimeval = irf_center_location_array[i]-irf_width-0.1
        f = interpolate.interp1d(
            (t - correcttimeval), 
            d[:, i],
            kind='linear', 
            bounds_error=False, 
            fill_value=0
        )
        d_corr[:, i] = f(t)
        
    return d_corr

def plot_heatmap(datasets, dataset_labels,
                 var_name="fitted_data",
                 plot_type="pcolormesh",
                 levels=20,
                 apply_chirp_correction=False,
                 zscale="symlog",
                 symlog_z_thresh=0.01,
                 vmin=None, vmax=None,
                 cmap=DEFAULT_CMAP,
                 yscale="linear",
                 symlog_y_thresh=1.0,
                 xlim=None, ylim=None,
                 invert_y=False,
                 layout='horizontal',
                 export=False):
    """
    Plots a 2D heatmap using Matplotlib directly, with layout and axis inversion.

    Args:
        ... (standard args) ...
        invert_y (bool): If True, inverts the y-axis (time). Defaults to False.
        layout (str): 'horizontal' (side-by-side) or 'vertical' (stacked)
                      subplot arrangement. Defaults to 'horizontal'.
    """
    
    # --- 1. Standardize inputs ---
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(dataset_labels, list):
        dataset_labels = [dataset_labels]
    
    if len(datasets) != len(dataset_labels):
        raise ValueError("The number of datasets must match the number of labels.")
    
    n_datasets = len(datasets)

    # --- NEW: Handle plot layout ---
    if layout == 'vertical':
        nrows, ncols = n_datasets, 1
        figsize = (8, 6 * n_datasets)  # (width, height)
    else:  # default to 'horizontal'
        nrows, ncols = 1, n_datasets
        figsize = (7 * n_datasets, 6)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False
    )

    # --- 2. Main loop over each dataset ---
    for ax, ds, label in zip(axes.flat, datasets, dataset_labels):
    
        try:
            # Get the data array and coordinates
            data_array = ds[var_name]
            X = ds['spectral'].values
            Y = ds['time'].values
            Z = data_array.values  # Z shape is (time, spectral)
            y_label = "Time (ps)"
            if export:
                    export_var = ds["data"].to_dataframe()
                    export_var.to_csv(label+"2d.csv")

            # --- 3. Handle Chirp Correction (Z-Data) ---
            if apply_chirp_correction:
                try:
                    irf_width = ds['irf_width'].values
                    chirp_array_1d = ds['irf_center_location'].values
                    Z = _apply_chirp_correction_to_data(
                        data_array, Y, X, chirp_array_1d.squeeze(), irf_width
                    )
                except KeyError:
                    print(f"Warning: 'irf_center_location' not found in '{label}'. Cannot apply chirp correction.")
                except Exception as e:
                    print(f"Warning: Failed to apply chirp correction for '{label}': {e}. Plotting uncorrected data.")
            
            # --- 4. Handle Color Scale (Z-Coordinate) ---
            if zscale == 'symlog':
                norm = mcolors.SymLogNorm(linthresh=symlog_z_thresh, vmin=vmin, vmax=vmax, base=10)
            else: # linear
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
            # --- 5. Plotting (using 1D X, 1D Y, 2D Z) ---
            if plot_type == 'contourf':
                h = ax.contourf(
                    X, Y, Z, levels=levels, cmap=cmap, norm=norm
                )
            else: # default to pcolormesh
                h = ax.pcolormesh(
                    X, Y, Z, cmap=cmap, norm=norm, shading='auto'
                )
            
            # --- 6. Add Colorbar Manually ---
            cbar_label = "$\Delta A$ (mOD)"
            cbar = fig.colorbar(h, ax=ax, label=cbar_label)
            for a in cbar.ax.get_yticklabels():
                a.set_fontsize(18)
            # --- 7. Format Axes ---
            ax.set_title(label, fontsize=14)
            ax.set_xlabel("Wavelength (nm)", fontsize=18)
            ax.set_ylabel(y_label, fontsize=18)

            if yscale == 'symlog':
                ax.set_yscale('symlog', linthresh=symlog_y_thresh)
            else:
                ax.set_yscale('linear')
            
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)

            # --- NEW: Invert Y-axis if requested ---
            if invert_y:
                ax.invert_yaxis()

        except KeyError:
            print(f"Warning: Variable '{var_name}' not found in '{label}'. Skipping.")
        except Exception as e:
            print(f"Error plotting '{label}': {e}")

    # --- 8. Final Touches ---
    #title = f"Heatmap of {var_name}"
    #if apply_chirp_correction:
    #    title += " (Chirp-Corrected Data)"
    #fig.suptitle(title, fontsize=20, y=1.03)
    plt.tight_layout()
    #format_publication_plot_no_latex(ax=ax)
    plt.show()

    return fig, axes