# Import functions to top level for easy access
from .Decomposition_plot import plot_decomposition
from .format_publication_plot_no_latex import format_publication_plot_no_latex
from .plot_heatmap import plot_heatmap
from .plot_difference_map import plot_difference_map
from .plot_multi_spectral_slices import plot_multi_spectral_slices
from .plot_multi_spectral_slices_animation import plot_multi_spectral_slices_animation
from .plot_multi_time_traces import plot_multi_time_traces
from .plot_species_associated_spectra import plot_species_associated_spectra
from .plot_time_traces import plot_fit_vs_data

# Expose submodules
from . import functions
from . import megacomplexes