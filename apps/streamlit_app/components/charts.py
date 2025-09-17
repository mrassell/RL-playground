"""
Chart components for Streamlit app.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd


def create_line_chart(
    data: List[float],
    title: str,
    xlabel: str = "Updates",
    ylabel: str = "Value",
    color: str = "blue",
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Create a simple line chart.
    
    Args:
        data: List of values to plot
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Line color
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(data))
    ax.plot(x, data, color=color, linewidth=2, marker='o', markersize=4)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    if data:
        y_min, y_max = min(data), max(data)
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    return fig


def create_multi_line_chart(
    data_dict: Dict[str, List[float]],
    title: str,
    xlabel: str = "Updates",
    ylabel: str = "Value",
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Create a multi-line chart.
    
    Args:
        data_dict: Dictionary mapping line names to data lists
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (name, data) in enumerate(data_dict.items()):
        if data:
            x = range(len(data))
            color = colors[i % len(colors)]
            ax.plot(x, data, color=color, linewidth=2, marker='o', markersize=4, label=name)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def create_bar_chart(
    data: Dict[str, float],
    title: str,
    xlabel: str = "Category",
    ylabel: str = "Value",
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Create a bar chart.
    
    Args:
        data: Dictionary mapping categories to values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    categories = list(data.keys())
    values = list(data.values())
    
    bars = ax.bar(categories, values, color=['blue', 'red', 'green', 'orange'])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def create_metrics_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Create a metrics table DataFrame.
    
    Args:
        metrics: Dictionary of metric names to values
        
    Returns:
        DataFrame with metrics
    """
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    df['Value'] = df['Value'].round(4)
    return df


def save_plot(fig: plt.Figure, filename: str, dpi: int = 300):
    """
    Save plot to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: DPI for saved image
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def create_training_curves(
    metrics_history: List[Dict[str, float]],
    metric_names: List[str],
    title: str = "Training Progress"
) -> plt.Figure:
    """
    Create training curves from metrics history.
    
    Args:
        metrics_history: List of metric dictionaries
        metric_names: List of metric names to plot
        title: Chart title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric_name in enumerate(metric_names[:4]):
        if i < len(axes):
            data = [m.get(metric_name, 0) for m in metrics_history]
            if data:
                x = range(len(data))
                axes[i].plot(x, data, 'b-', linewidth=2, marker='o', markersize=4)
                axes[i].set_title(metric_name.replace('_', ' ').title(), fontweight='bold')
                axes[i].set_xlabel('Updates')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(metric_names), 4):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig
