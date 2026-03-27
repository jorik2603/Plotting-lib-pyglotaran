import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

def estimate_chirp_sustained_edge(dataset, v, t_range=(-1, 8), wv_range=None, poly_order=3, c_wl = 470, dt = 25,
                                  cmap='seismic', t_delta=0.2, threshold= 0.1, edge_width_fs=200):
    """
    Finds the chirp by tracking the START of a sustained rising or falling edge (> 200fs).
    
    t_delta: Max allowable time jump between adjacent wavelengths.
    edge_width_fs: The duration in femtoseconds the edge must be sustained.
    """
    A = dataset.data
    full_t = A.time.to_numpy()
    wv = A.spectral.to_numpy()
    d = A.to_numpy() # Shape: (time, spectral)

    if wv_range is not None:
        wv_mask = (wv >= wv_range[0]) & (wv <= wv_range[1])
        wv = wv[wv_mask]
        d = d[:, wv_mask]
        
    # 1. Determine temporal resolution dynamically
    edge_width_ps = edge_width_fs / 1000.0
    window_size_indices = max(1, int(edge_width_ps / dt))

    detected_t0 = []
    valid_wv = []
    
    # 2. Initialize tracking at the first wavelength
    t_mask = (full_t >= t_range[0]) & (full_t <= t_range[1])
    search_t = full_t[t_mask]
    
    first_trace = d[t_mask, 0]
    smoothed_init = savgol_filter(first_trace, 11, 2)
    grad_init = np.gradient(smoothed_init)
    
    # Find the peak of the sustained gradient
    sustained_init = uniform_filter1d(grad_init, size=window_size_indices)
    idx_max_init = np.argmax(np.abs(sustained_init))
    
    # Backtrack to find the start of the edge (10% threshold)
    peak_grad_mag_init = np.abs(grad_init[idx_max_init])
    threshold_grad_init = 0.10 * peak_grad_mag_init
    
    idx_start_init = idx_max_init
    while idx_start_init > 0 and np.abs(grad_init[idx_start_init]) > threshold_grad_init:
        # Stop if the slope changes direction
        if grad_init[idx_start_init] * grad_init[idx_max_init] < 0:
            break
        idx_start_init -= 1
        
    current_t0 = search_t[idx_start_init]

    # 3. Iterate through wavelengths with tracking window
    for i in range(d.shape[1]):
        # Define search window based strictly on the last found point
        window_min = current_t0 - t_delta
        window_max = current_t0 + t_delta
        
        local_mask = (full_t >= window_min) & (full_t <= window_max)
        
        # If the window is too small for SavGol, reuse the last good point
        if np.sum(local_mask) < 12:
            detected_t0.append(current_t0)
            valid_wv.append(wv[i])
            continue
            
        local_t = full_t[local_mask]
        local_trace = d[local_mask, i]
        
        # Smooth and calculate gradient
        smoothed = savgol_filter(local_trace, 11, 2)
        grad = np.gradient(smoothed)
        
        # Find the peak of the edge
        sustained_grad = uniform_filter1d(grad, size=window_size_indices)
        idx_max = np.argmax(np.abs(sustained_grad))
        
        # Backtrack to find the start of the edge within the local window
        peak_grad_mag = np.abs(grad[idx_max])
        threshold_grad = threshold * peak_grad_mag
        
        idx_start = idx_max
        while idx_start > 0 and np.abs(grad[idx_start]) > threshold_grad:
            if grad[idx_start] * grad[idx_max] < 0:
                break
            idx_start -= 1
            
        # Update trackers
        current_t0 = local_t[idx_start]
        
        detected_t0.append(current_t0)
        valid_wv.append(wv[i])

    # 4. Polynomial Fit
    detected_t0 = np.array(detected_t0)
    valid_wv = np.array(valid_wv)
    wl_norm = (valid_wv - c_wl) / 100
    z = np.polyfit(wl_norm, detected_t0, poly_order)
    p = np.poly1d(z)

    # 5. Visualization
    fig, ax = plt.subplots(figsize=(14, 7))
    mesh = ax.pcolormesh(full_t, wv, d.T, shading='nearest', cmap=cmap, vmin=v[0], vmax=v[1])
    
    # Plot detected points and the resulting fit
    ax.scatter(detected_t0, valid_wv, color='black', s=2, alpha=0.5, label='Detected Edge Starts')
    
    fit_wv = np.linspace(wv.min(), wv.max(), 100)
    ax.plot(p((fit_wv - c_wl)/100), fit_wv, color='black', lw=2.5, linestyle='--', label='Poly Fit')
    
    ax.set_xlim(t_range[0] - 0.5, t_range[1] + 1.0)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Wavelength (nm)')
    ax.set_title(f"Sustained Edge Start Tracking (>{edge_width_fs}fs, Free Movement)")
    plt.colorbar(mesh, label='$\Delta$A')
    plt.legend()
    plt.show()
    
    print(z)
    return z, np.column_stack((detected_t0, valid_wv))