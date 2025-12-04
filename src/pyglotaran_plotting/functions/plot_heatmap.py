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
    print("Warning: 'cmcrameri' library not found. Falling back to 'RdBu_r' colormap.")
    DEFAULT_CMAP = 'RdBu_r'

def _apply_chirp_correction_to_data(data_array, time_coords, spectral_coords, 
                                    irf_center_location_array, irf_width):
    """
    Applies chirp correction to a 2D data array using 1D interpolation.

    This shifts the data along the time axis for each wavelength based on the 
    IRF center location (the "chirp"), effectively aligning time-zero across 
    all wavelengths.

    Parameters
    ----------
    data_array : xarray.DataArray
        The 2D data (Time x Spectral).
    time_coords : numpy.ndarray
        Array of time coordinate values.
    spectral_coords : numpy.ndarray
        Array of spectral coordinate values.
    irf_center_location_array : numpy.ndarray
        Array containing the IRF center location (time-zero) for each spectral point.
    irf_width : float
        The width of the IRF, used as an offset.

    Returns
    -------
    numpy.ndarray
        The chirp-corrected 2D data array.

    Raises
    ------
    ValueError
        If the length of the chirp data does not match the spectral coordinates.
    """
    t = time_coords
    wv = spectral_coords
    d = data_array.values  # Shape: (time, spectral)
    d_corr = np.zeros_like(d)

    if len(wv) != len(irf_center_location_array):
        raise ValueError(
            f"Chirp data length ({len(irf_center_location_array)}) does not match "
            f"spectral coordinate length ({len(wv)})."
        )

    for i in range(len(wv)):
        # Calculate the corrected time origin
        correcttimeval = irf_center_location_array[i] - irf_width - 0.1
        
        # Interpolate data to the new time axis
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
                 normalize=False,
                 measurement_type="TA"):
    """
    Plots 2D heatmaps for one or multiple datasets.

    Supports chirp correction, various scaling options (symlog/linear), and 
    flexible layouts.

    Parameters
    ----------
    datasets : list or xarray.Dataset
        Single dataset or list of datasets to plot.
    dataset_labels : list or str
        Labels corresponding to the datasets.
    var_name : str, optional
        Variable to plot (e.g., 'fitted_data', 'data'). Default is 'fitted_data'.
    plot_type : str, optional
        'pcolormesh' (heatmap) or 'contourf' (filled contours). Default is 'pcolormesh'.
    levels : int, optional
        Number of levels for contourf plots. Default is 20.
    apply_chirp_correction : bool, optional
        If True, visually corrects for the IRF chirp (time-zero dispersion). Default is False.
    zscale : str, optional
        Scaling for the color axis ('linear' or 'symlog'). Default is 'symlog'.
    symlog_z_thresh : float, optional
        Linear threshold for symlog Z scaling. Default is 0.01.
    vmin, vmax : float, optional
        Min/Max values for color scaling.
    cmap : Colormap, optional
        Matplotlib colormap. Default is cmc.vik or 'RdBu_r'.
    yscale : str, optional
        Y-axis scaling ('linear' or 'symlog'). Default is 'linear'.
    symlog_y_thresh : float, optional
        Linear threshold for symlog Y scaling. Default is 1.0.
    xlim, ylim : tuple, optional
        Limits for x (spectral) and y (time) axes.
    invert_y : bool, optional
        If True, inverts the Y-axis. Default is False.
    layout : str, optional
        'horizontal' or 'vertical' arrangement of subplots. Default is 'horizontal'.
    normalize : bool, optional
        If True, normalizes data to the maximum value. Default is False.
    measurement_type : str, optional
        'TA' (Transient Absorption) or other. Affects colorbar label. Default is "TA".

    Returns
    -------
    tuple
        (fig, axes) The figure and array of axes objects.
    """
    # Standardize inputs to lists
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(dataset_labels, list):
        dataset_labels = [dataset_labels]
    
    if len(datasets) != len(dataset_labels):
        raise ValueError("The number of datasets must match the number of labels.")
    
    n_datasets = len(datasets)

    # Setup layout
    if layout == 'vertical':
        nrows, ncols = n_datasets, 1
        figsize = (8, 6 * n_datasets)
    else:
        nrows, ncols = 1, n_datasets
        figsize = (7 * n_datasets, 6)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False
    )

    # Main loop over datasets
    for ax, ds, label in zip(axes.flat, datasets, dataset_labels):
        try:
            data_array = ds[var_name]
            X = ds['spectral'].values
            Y = ds['time'].values
            Z = data_array.values  # Shape: (time, spectral)
            y_label = "Time (ps)"
            
            # Apply Chirp Correction
            if apply_chirp_correction:
                try:
                    irf_width = ds['irf_width'].values
                    chirp_array_1d = ds['irf_center_location'].values
                    Z = _apply_chirp_correction_to_data(
                        data_array, Y, X, chirp_array_1d.squeeze(), irf_width
                    )
                except KeyError:
                    print(f"Warning: 'irf_center_location' not found in '{label}'. Skipping chirp correction.")
                except Exception as e:
                    print(f"Warning: Chirp correction failed for '{label}': {e}. Plotting uncorrected data.")
            
            # Configure Color Scale
            if zscale == 'symlog':
                norm = mcolors.SymLogNorm(linthresh=symlog_z_thresh, vmin=vmin, vmax=vmax, base=10)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                
            if normalize:                  
                norm_val = np.max(np.abs(Z))
                if norm_val != 0:
                    Z = Z / norm_val
                            
            # Plot Data
            if plot_type == 'contourf':
                h = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
            else:
                h = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='auto')
            
            # Add Colorbar
            cbar_label = "$\Delta A$ (mOD)" if measurement_type == "TA" else "I (A.U.)"
            cbar = fig.colorbar(h, ax=ax, label=cbar_label)
            for a in cbar.ax.get_yticklabels():
                a.set_fontsize(18)

            # Format Axes
            ax.set_title(label, fontsize=14)
            ax.set_xlabel("Wavelength (nm)", fontsize=18)
            ax.set_ylabel(y_label, fontsize=18)

            if yscale == 'symlog':
                ax.set_yscale('symlog', linthresh=symlog_y_thresh)
            else:
                ax.set_yscale('linear')
            
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)
            if invert_y: ax.invert_yaxis()

        except KeyError:
            print(f"Warning: Variable '{var_name}' not found in '{label}'. Skipping.")
        except Exception as e:
            print(f"Error plotting '{label}': {e}")

    plt.tight_layout()
    plt.show()

    return fig, axes