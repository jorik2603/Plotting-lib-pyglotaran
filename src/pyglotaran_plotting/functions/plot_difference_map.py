def calculate_difference_dataset(ds_target, ds_reference, var_name="fitted_data", apply_chirp=True):
    """
    Calculates the difference between two datasets (Target - Reference).
    1. Apply Chirp Correction to both (on original grids).
    2. Interpolate Reference to match Target's Time/Spectral coordinates.
    3. Subtract.
    """
    # Create working copies so we don't modify the originals
    ds_t = ds_target.copy(deep=True)
    ds_r = ds_reference.copy(deep=True)

    # --- 1. Apply Chirp Correction (Independently on original grids) ---
    if apply_chirp:
        try:
            # Correct Target
            t_data = ds_t[var_name].values
            t_time = ds_t['time'].values
            t_spec = ds_t['spectral'].values
            t_irf_w = ds_t['irf_width'].values
            t_irf_c = ds_t['irf_center_location'].values
            
            ds_t[var_name].values = _apply_chirp_correction_to_data(
                ds_t[var_name], t_time, t_spec, t_irf_c.squeeze(), t_irf_w
            )

            # Correct Reference
            r_data = ds_r[var_name].values
            r_time = ds_r['time'].values
            r_spec = ds_r['spectral'].values
            r_irf_w = ds_r['irf_width'].values
            r_irf_c = ds_r['irf_center_location'].values

            ds_r[var_name].values = _apply_chirp_correction_to_data(
                ds_r[var_name], r_time, r_spec, r_irf_c.squeeze(), r_irf_w
            )
            print("Chirp correction applied to both datasets independently.")

        except KeyError as e:
            print(f"Warning: Missing IRF params ({e}). Proceeding with raw subtraction.")

    # --- 2. Align Grids (Fixing the Shape Mismatch) ---
    # This uses xarray to interpolate ds_r onto the coordinates of ds_t
    # If ds_t is (871, 205) and ds_r is (871, 208), ds_r_aligned becomes (871, 205)
    ds_r_aligned = ds_r.interp_like(ds_t, method="linear")

    # --- 3. Calculate Difference ---
    # Now both have the exact same shape and coordinates
    diff_matrix = ds_t[var_name] - ds_r_aligned[var_name]
    
    # Store result in the target structure
    ds_diff = ds_t.copy(deep=True)
    ds_diff[var_name] = diff_matrix
    
    return ds_diff

def plot_difference_map(ds_target, ds_reference, label, 
                        var_name="fitted_data", 
                        apply_chirp_correction=True,
                        vmin=None, vmax=None,
                        auto_center_color=True,
                        **kwargs):
    """
    Wrapper to calculate difference and plot it immediately using plot_heatmap.
    
    Args:
        ds_target: The main dataset.
        ds_reference: The dataset to subtract (Target - Reference).
        label: Label for the plot title.
        auto_center_color: If True, calculates symmetric vmin/vmax to center 0 at white/grey.
        **kwargs: Arguments passed to plot_heatmap.
    """
    
    # 1. Generate the difference dataset
    ds_diff = calculate_difference_dataset(
        ds_target, 
        ds_reference, 
        var_name=var_name, 
        apply_chirp=apply_chirp_correction
    )
    
    # 2. Handle visual scaling (Difference maps look best when symmetric around 0)
    if auto_center_color and vmin is None and vmax is None:
        data_max = np.nanmax(np.abs(ds_diff[var_name].values))
        vmax = data_max
        vmin = -data_max
        print(f"Auto-centering colormap: vmin={vmin:.2f}, vmax={vmax:.2f}")

    # 3. Plot using the existing function
    # Note: We set apply_chirp_correction=False because we already did it in step 1.
    plot_heatmap(
        datasets=[ds_diff],
        dataset_labels=[f"{label}"],
        var_name=var_name,
        apply_chirp_correction=False, 
        vmin=vmin,
        vmax=vmax,
        **kwargs
    )