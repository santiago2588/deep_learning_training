from typing import List, Optional
from matplotlib.axes import Axes
from utils.plotting.fonts import load_font

__all__ = [
    'format_tick_label',
    'format_axis_ticks',
    'format_spines',
    'make_fig_pretty'
]

PATTERNS = [
    "/", "\\", "|", "-", "o", "O", ".", "*", "+", "x", "X", ":", "=", " ", " "
]

def format_tick_label(label: str) -> str:
    """
    Format tick label text to uppercase if it's a string, leave numbers unchanged.
    
    Args:
        label (str): The tick label to format
        
    Returns:
        str: Formatted tick label
    """
    try:
        if str(label).replace(".", "").replace("-", "").isdigit():
            return str(label)
        return str(label).upper()
    except AttributeError:
        return ""

def format_axis_ticks(
    ax: Axes,
    axis: str,
    share_axis: bool,
    is_3d: bool,
    font_manager,
    tick_size: int
) -> None:
    """
    Format ticks for a given axis (x, y, or z).
    
    Args:
        ax: Matplotlib axis object
        axis: Which axis to format ('x', 'y', or 'z')
        share_axis: Whether axis is shared
        is_3d: Whether plot is 3D
        font_manager: Font manager instance
        tick_size: Font size for tick labels
    """
    # Set ticks
    getattr(ax, f'set_{axis}ticks')(getattr(ax, f'get_{axis}ticks')())
    
    # Handle shared axes
    if share_axis and not is_3d and axis in ['x', 'y']:
        getattr(ax, f'set_{axis}ticklabels')([''] * len(getattr(ax, f'get_{axis}ticklabels')()))
        
    # Format labels if they exist
    labels = [label.get_text() for label in getattr(ax, f'get_{axis}ticklabels')()]
    if all(labels):
        labels_fmt = [format_tick_label(label) for label in labels]
        getattr(ax, f'set_{axis}ticklabels')(labels_fmt)
        
    # Apply font properties
    for label in getattr(ax, f'get_{axis}ticklabels')():
        label.set_fontproperties(font_manager)
        label.set_fontsize(tick_size)

def format_spines(ax: Axes, ylabels_fmt: Optional[List[str]], sharey: bool, timestamp: bool, tufte_style: bool = False) -> None:
    """
    Format plot spines (borders) following Tufte principles when enabled.
    
    Args:
        ax: Matplotlib axis object
        ylabels_fmt: Formatted y-axis labels
        sharey: Whether y-axis is shared
        timestamp: Whether plot includes timestamp
        tufte_style: Whether to apply Tufte-style minimalist formatting
    """
    if tufte_style:
        # In Tufte style, remove all spines except where needed for data context
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(False)
        
        # Add subtle ticks without lines
        ax.tick_params(axis='both', which='both', length=3, width=0.5, 
                       colors='gray', labelcolor='black', pad=3)
        return
        
    # Standard formatting (non-Tufte)
    # Remove top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    if timestamp:
        return
        
    # Set spine bounds
    x_bounds = ax.get_xticks()
    y_bounds = ax.get_yticks()
    
    ax.spines['bottom'].set_bounds(min(x_bounds), max(x_bounds) * 1.025)
    
    # Handle y-axis spine bounds based on data characteristics
    if (ylabels_fmt or sharey) and not (ylabels_fmt[0].replace(".", "").isdigit()):
        ax.spines['left'].set_bounds(
            min(y_bounds) + 0.025 * min(y_bounds),
            max(y_bounds) * 1.15
        )
    elif min(y_bounds) < 0:
        ax.spines['left'].set_bounds(min(y_bounds) * 0.85, max(y_bounds) * 1.05)
        ax.set_yticks(ax.get_yticks()[1:])
    else:
        ax.spines['left'].set_bounds(
            min(y_bounds) * (1.05 if min(y_bounds) == 0 else 1.025),
            max(y_bounds) * 1.025
        )
        ax.set_yticks(ax.get_yticks()[1:])

def make_fig_pretty(
    ax: Axes,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    ctab: bool = False,
    title: str = "",
    legend: bool = True,
    legd_title: str = "",
    legd_loc: str = "best",
    legd_labels: Optional[List[str]] = None,
    sharex: bool = False,
    sharey: bool = False,
    timestamp: bool = False,
    xtick_fsize: int = 11,
    ytick_fsize: int = 11,
    xlabel_fsize: int = 11,
    ylabel_fsize: int = 11,
    title_fsize: int = 12,
    grid: bool = False,
    tufte_style: bool = False,
    background_color: Optional[str] = None,
    range_frame: bool = False
) -> None:
    """
    Apply consistent formatting to a matplotlib figure with Tufte principles option.
    
    Args:
        ax: Matplotlib axis object
        xlabel: X-axis label
        ylabel: Y-axis label
        zlabel: Z-axis label (for 3D plots)
        ctab: Whether to add patterns to cross-tables
        title: Plot title
        legend: Whether to show legend
        legd_title: Legend title
        legd_loc: Legend location
        legd_labels: Custom legend labels
        sharex: Whether x-axis is shared
        sharey: Whether y-axis is shared
        timestamp: Whether to include timestamp
        xtick_fsize: X-axis tick font size
        ytick_fsize: Y-axis tick font size
        xlabel_fsize: X-axis label font size
        ylabel_fsize: Y-axis label font size
        title_fsize: Title font size
        grid: Whether to show grid
        tufte_style: Whether to apply Tufte-style minimalist formatting
        background_color: Optional custom background color
        range_frame: Whether to use a range frame instead of full axes (Tufte-style)
    """
    is_3d = "3D" in str(type(ax.axes))
    
    # Load and configure fonts
    fm = load_font()
    fm.set_size(9)
    
    # Set background color if specified
    if background_color:
        ax.set_facecolor(background_color)
        
    # Set title and axis labels
    if tufte_style:
        # In Tufte style, title should be less prominent - smaller and not all caps
        ax.set_title(title, fontproperties=fm, fontsize=title_fsize-1, color='#505050')
        ax.set_xlabel(xlabel, fontproperties=fm, fontsize=xlabel_fsize, color='#505050')
        ax.set_ylabel(ylabel, fontproperties=fm, fontsize=ylabel_fsize, color='#505050')
        
        if is_3d:
            ax.set_zlabel(zlabel, fontproperties=fm, fontsize=11, color='#505050')
    else:
        # Standard formatting
        ax.set_title(" ".join(title.upper()), fontproperties=fm, fontsize=title_fsize)
        ax.set_xlabel(" ".join(xlabel.upper()), fontproperties=fm, fontsize=xlabel_fsize)
        ax.set_ylabel(" ".join(ylabel.upper()), fontproperties=fm, fontsize=ylabel_fsize)
        
        if is_3d:
            ax.set_zlabel(" ".join(zlabel.upper()), fontproperties=fm, fontsize=11)
    
    # Format axis ticks
    format_axis_ticks(ax, 'x', sharex, is_3d, fm, xtick_fsize)
    format_axis_ticks(ax, 'y', sharey, is_3d, fm, ytick_fsize)
    if is_3d:
        format_axis_ticks(ax, 'z', False, is_3d, fm, 11)
    
    # Format spines
    ylabels = [label.get_text() for label in ax.get_yticklabels()]
    ylabels_fmt = [format_tick_label(label) for label in ylabels] if all(ylabels) else None
    format_spines(ax, ylabels_fmt, sharey, timestamp, tufte_style)
    
    # Apply range frame (Tufte-style) if requested
    if tufte_style and range_frame and not is_3d:
        # Get data range
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Create minimal "range frame" - only show axes at the data range
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_bounds(x_min, x_max)
        ax.spines['left'].set_bounds(y_min, y_max)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
    
    # Handle cross-table patterns
    if ctab and "patches" in str(ax.get_children()):
        # Get number of unique plots by counting legend entries
        handles, labels = ax.get_legend_handles_labels()
        n_plots = len(labels) if labels else 1
        
        # Count total bars
        n_bars = sum(1 for x in ax.get_children() if "Rectangle" in str(x)) - 1
        
        # Calculate elements per plot
        len_elems = n_bars // n_plots if n_plots > 0 else n_bars
        
        if len_elems > 0:
            hatches = [p for p in PATTERNS for _ in range(len_elems)]
            for bar, hatch in zip(ax.patches, hatches):
                bar.set_hatch(hatch)
    
    # Check if a legend already exists
    existing_legend = ax.get_legend()
    
    # Add or format legend if requested
    if legend:
        # If there's already a legend, format it
        if existing_legend is not None:
            # Format existing legend
            existing_legend.set_title(legd_title.upper() if not tufte_style else legd_title)
            for text in existing_legend.get_texts():
                text.set_fontproperties(fm)
                text.set_fontsize(11)
                if not tufte_style:
                    text.set_text(text.get_text().upper())
            
            existing_legend.get_title().set_fontproperties(fm)
            
            # For Tufte style, make legend more subtle
            if tufte_style:
                existing_legend.get_frame().set_facecolor('white')
                existing_legend.get_frame().set_linewidth(0)
        else:
            # Create new legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                if tufte_style:
                    legend_labels = legd_labels if legd_labels else labels
                    _legend = ax.legend(
                        handles,
                        legend_labels,
                        title=legd_title,
                        prop=fm,
                        fontsize=11,
                        loc=legd_loc,
                        frameon=False
                    )
                else:
                    legend_labels = legd_labels if legd_labels else [x.upper() for x in labels]
                    _legend = ax.legend(
                        handles,
                        legend_labels,
                        title=legd_title.upper(),
                        prop=fm,
                        fontsize=11,
                        loc=legd_loc
                    )
                _legend.get_title().set_fontproperties(fm)
    
    # Show grid if requested
    if grid:
        if tufte_style:
            # Tufte prefers subtle, lighter grid lines
            ax.grid(color="#E0E0E0", linestyle=":", linewidth=0.3)
        else:
            ax.grid(color="gray", linestyle="--", linewidth=0.5)