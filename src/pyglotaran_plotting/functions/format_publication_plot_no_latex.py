import matplotlib.pyplot as plt
import numpy as np

def format_publication_plot_no_latex(ax,
                                     title=None,
                                     xlabel=None,
                                     ylabel=None,
                                     title_fontsize=14,
                                     label_fontsize=18,
                                     tick_fontsize=18,
                                     legend_fontsize=18,
                                     font_family='serif',
                                     show_legend=True,
                                     legend_loc='best',
                                     line_width=1.5,
                                     grid_alpha=0.2,
                                     tick_direction='in',
                                     spine_color='black'):
    """
    Formats a matplotlib Axes object for scientific publication without requiring LaTeX.

    This function applies a cleaner style using Matplotlib's 'mathtext' engine for
    mathematical expressions and STIX fonts, which mimic the look of LaTeX documents.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to format.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis. Supports mathtext (e.g., r'$\\alpha_{lim}$').
    ylabel : str, optional
        The label for the y-axis. Supports mathtext.
    title_fontsize : int, optional
        Font size for the title. Default is 14.
    label_fontsize : int, optional
        Font size for axis labels. Default is 18.
    tick_fontsize : int, optional
        Font size for tick labels. Default is 18.
    legend_fontsize : int, optional
        Font size for the legend. Default is 18.
    font_family : str, optional
        The font family to use ('serif', 'sans-serif'). Default is 'serif'.
    show_legend : bool, optional
        Whether to display the legend. Default is True.
    legend_loc : str, optional
        The location code for the legend. Default is 'best'.
    line_width : float, optional
        The width for plotted lines. Default is 1.5.
    grid_alpha : float, optional
        Transparency of the grid. Set to 0 for no grid. Default is 0.2.
    tick_direction : str, optional
        Direction of axis ticks ('in', 'out', 'inout'). Default is 'in'.
    spine_color : str, optional
        Color of the plot's border spines. Default is 'black'.

    Returns
    -------
    None
        Modifies the provided Axes object in-place.
    """
    # Set font properties for a professional look (STIX fonts)
    plt.rcParams['font.family'] = font_family
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize, weight='bold')

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, direction=tick_direction)
    ax.tick_params(axis='both', which='minor', direction=tick_direction)

    # Customize spines (cleaner borders)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(1)

    # Customize plotted lines
    for line in ax.get_lines():
        line.set_linewidth(line_width)

    # Customize legend
    if show_legend:
        # Check if there are labeled handles to create a legend for
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=False)

    # Add grid
    if grid_alpha > 0:
        ax.grid(True, which='major', linestyle='--', alpha=grid_alpha)

    # Ensure tight layout
    fig = ax.get_figure()
    if fig:
        fig.tight_layout()