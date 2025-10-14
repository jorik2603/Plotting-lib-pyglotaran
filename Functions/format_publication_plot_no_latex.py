import matplotlib.pyplot as plt
import numpy as np

def format_publication_plot_no_latex(ax,
                                     title=None,
                                     xlabel=None,
                                     ylabel=None,
                                     title_fontsize=14,
                                     label_fontsize=12,
                                     tick_fontsize=10,
                                     legend_fontsize=10,
                                     font_family='serif',
                                     show_legend=True,
                                     legend_loc='best',
                                     line_width=1.5,
                                     grid_alpha=0.2,
                                     tick_direction='in',
                                     spine_color='black'):
    """
    Formats a matplotlib Axes object for a scientific publication
    without requiring a LaTeX installation.

    This function uses Matplotlib's built-in font handling and 'mathtext'
    engine for mathematical expressions.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to format.
        title (str, optional): The title of the plot. Defaults to None.
        xlabel (str, optional): The label for the x-axis. Supports mathtext
                                (e.g., r'$\\alpha_{lim}$'). Defaults to None.
        ylabel (str, optional): The label for the y-axis. Supports mathtext.
                                Defaults to None.
        title_fontsize (int, optional): Font size for the title. Defaults to 14.
        label_fontsize (int, optional): Font size for axis labels. Defaults to 12.
        tick_fontsize (int, optional): Font size for tick labels. Defaults to 10.
        legend_fontsize (int, optional): Font size for the legend. Defaults to 10.
        font_family (str, optional): The font family to use ('serif', 'sans-serif').
                                     Defaults to 'serif'.
        show_legend (bool, optional): Whether to display the legend.
                                      Defaults to True.
        legend_loc (str, optional): The location of the legend.
                                    Defaults to 'best'.
        line_width (float, optional): The width for plotted lines. Defaults to 1.5.
        grid_alpha (float, optional): Transparency of the grid. Set to 0 for no grid.
                                      Defaults to 0.2.
        tick_direction (str, optional): Direction of axis ticks ('in', 'out', 'inout').
                                        Defaults to 'in'.
        spine_color (str, optional): Color of the plot's border spines.
                                     Defaults to 'black'.
    """
    # 1. Set font properties for a professional look
    # Uses STIX fonts which are designed for scientific publishing
    plt.rcParams['font.family'] = font_family
    plt.rcParams['mathtext.fontset'] = 'stix' # Or 'cm' for Computer Modern look

    # 2. Set labels and title with specified font sizes
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize, weight='bold')

    # 3. Customize ticks
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=tick_fontsize,
                   direction=tick_direction)
    ax.tick_params(axis='both',
                   which='minor',
                   direction=tick_direction)

    # 4. Customize spines (the plot border) for a cleaner look
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(1)

    # 5. Customize plotted lines
    for line in ax.get_lines():
        line.set_linewidth(line_width)

    # 6. Customize legend
    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=False)

    # 7. Add a subtle grid
    if grid_alpha > 0:
        ax.grid(True, which='major', linestyle='--', alpha=grid_alpha)

    # 8. Ensure tight layout to prevent labels from overlapping
    fig = ax.get_figure()
    fig.tight_layout()