# pyglotaran-plotting

**pyglotaran-plotting** is a Python library designed to facilitate the visualization of data and results from [pyglotaran](https://github.com/glotaran/pyglotaran), a tool for global analysis of time-resolved spectroscopy data. It provides high-level plotting functions for Transient Absorption (TA) and Time-Resolved Photoluminescence (TRPL) datasets, along with custom model components.

## Features

* **2D Heatmaps**: Visualize data matrices with support for chirp correction, symplectic log scaling, and difference maps.
* **Trace Comparisons**: Plot kinetic traces or spectral slices across multiple datasets to compare experimental conditions.
* **Decomposition Plots**: Visualize the contributions of specific model components (e.g., Coherent Artifacts, Damped Oscillations) to a kinetic trace.
* **Publication Ready**: Includes formatting utilities to style plots for scientific publications without requiring LaTeX.
* **Custom Megacomplex**: Includes a `DelayDampedOscillationMegacomplex` for advanced modeling.

## Installation

### Prerequisites

* Python >= 3.9
* pyglotaran
* pyglotaran-extras
* matplotlib
* numpy
* xarray
* scipy
* cmcrameri (for perceptually uniform colormaps)

### Install from Source

You can install the package directly from the repository:

```bash
pip install git+https://github.com/jorik2603/plotting-lib-pyglotaran.git
```

Or, if you have cloned the repository locally:

```bash
cd plotting-lib-pyglotaran
pip install .
```

## Usage examples

First, import the library:

```python
import pyglotaran_plotting as pyp
```

### 1. Plotting Heatmaps

```python
# 'results' is a list of your pyglotaran result datasets
pyp.plot_heatmap(
    datasets=[result_dataset1, result_dataset2],
    dataset_labels=["Experiment A", "Experiment B"],
    var_name="fitted_data",     # Variable to plot
    apply_chirp_correction=True, # Apply chirp correction to visualization
    zscale="symlog",            # Use symplectic log scale for Z-axis
    symlog_z_thresh=0.01,
    layout="horizontal"         # 'horizontal' or 'vertical'
)
```

### 2. Difference Maps

Calculate and plot the difference between two datasets (Target - Reference). This automatically handles grid alignment and independent chirp correction before subtraction.

```python
pyp.plot_difference_map(
    ds_target=dataset_pump_probe,
    ds_reference=dataset_pump_only,
    label="Pump-Probe Difference",
    var_name="fitted_data"
)
```

### 3. Comparing Time Traces

Plot kinetic traces at specific wavelengths across multiple datasets.

```python
pyp.plot_multi_time_traces(
    datasets=[res1, res2],
    dataset_labels=["Sample 1", "Sample 2"],
    spectral_values=[500, 620, 750], # Wavelengths in nm
    measurement_type="TA",           # "TA" or "TRPL" (handles t0 logic)
    apply_chirp_correction=True,
    normalize=False
)
```

### 4. Comparing Spectral Slices

Plot spectra at specific time delays.

```python
pyp.plot_multi_spectral_slices(
    datasets=[res1],
    dataset_labels=["Sample 1"],
    time_values=[0.5, 10, 100],      # Time points in ps
    measurement_type="TA",
    apply_chirp_correction=True
)
```

### 5. Plotting Species Associated Spectra (SAS)

Plot the spectra associated with specific species defined in your model.

```python
pyp.plot_species_associated_spectra(
    datasets=[res1],
    dataset_labels=["Global Fit"],
    species_to_plot=["species_1", "species_2"]
)
```

### 6. Decomposition Plot

Analyze a specific trace by breaking it down into its model components (e.g., separating the coherent artifact or damped oscillations from the main decay).

```python
# Requires the full result object, not just the dataset
pyp.plot_decomposition(
    result=full_result_object,
    dataname="dataset_name",
    wavelength=550
)
```

### 7. Advanced Modeling: Delayed Damped Oscillation

This library includes a custom megacomplex DelayDampedOscillationMegacomplex. To use it in your pyglotaran models, you must register it first.

```python
from glotaran.plugin_system.megacomplex_registration import register_megacomplex
from pyglotaran_plotting.megacomplexes import DelayDampedOscillationMegacomplex

register_megacomplex(
    megacomplex_type='delayed-damped-oscillation',
    megacomplex=DelayDampedOscillationMegacomplex
)
```

You can then refer to type: delayed-damped-oscillation in your model YAML definition.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
