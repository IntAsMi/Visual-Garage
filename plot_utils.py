import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np

# --- Plot Configuration Options ---
PLOT_TYPES = {
    "Scatter Plot": "scatter",
    "Line Plot": "line",
    "Bar Chart": "bar",
    "Histogram": "histogram",
    "Box Plot": "box",
    "Violin Plot": "violin",
    "Density Heatmap": "density_heatmap",
    "Pie Chart": "pie",
    "Sunburst Chart": "sunburst",
    "Treemap": "treemap",
    # "3D Scatter": "scatter_3d", # Requires different arg handling
}

PLOT_ARGUMENTS = {
    "common": ["color", "facet_row", "facet_col", "hover_name", "hover_data", "custom_data", "title"],
    "scatter": ["x", "y", "size", "symbol", "trendline"],
    "line": ["x", "y", "line_group", "markers", "line_dash"],
    "bar": ["x", "y", "orientation", "barmode", "text"], # barmode: 'group', 'stack', 'relative'
    "histogram": ["x", "y", "nbins", "histfunc", "cumulative", "barnorm"], # histfunc: 'count', 'sum', 'avg', 'min', 'max'
    "box": ["x", "y", "points", "notched"], # points: 'all', 'outliers', 'suspectedoutliers', False
    "violin": ["x", "y", "box", "points"],
    "density_heatmap": ["x", "y", "z", "histfunc", "nbinsx", "nbinsy"],
    "pie": ["names", "values"],
    "sunburst": ["names", "parents", "values", "path"], # path is an alternative to names/parents
    "treemap": ["names", "parents", "values", "path"],
    # "scatter_3d": ["x", "y", "z", "size", "symbol"],
}

def get_plot_arguments(plot_type_value):
    args = PLOT_ARGUMENTS["common"][:]
    if plot_type_value in PLOT_ARGUMENTS:
        args.extend(PLOT_ARGUMENTS[plot_type_value])
    return list(set(args)) # Unique arguments

def estimate_plot_time(df_rows, num_dimensions, plot_type):
    """
    Very basic heuristic for plot time estimation.
    Returns estimated time in seconds.
    """
    if df_rows == 0:
        return 0

    base_factor = 0.000005  # Adjust based on typical machine performance
    
    # Complexity factors for different plot types
    plot_complexity_factor = {
        "scatter": 1, "line": 1, "bar": 0.8, "histogram": 0.7,
        "box": 0.5, "violin": 0.6, "pie": 0.3,
        "density_heatmap": 1.5, "sunburst": 2, "treemap": 2,
        "scatter_3d": 2.5
    }.get(plot_type, 1)

    # Dimension factor (more dimensions for color, size, facets increase complexity)
    dim_factor = 1 + (num_dimensions -1) * 0.3 # Assuming 1 dim is min (e.g. x for hist)

    estimated_time = df_rows * dim_factor * plot_complexity_factor * base_factor
    
    # Cap estimation for sanity
    return min(estimated_time, 300) # Don't estimate more than 5 minutes

def create_plotly_figure(df: pd.DataFrame, plot_type_value: str, plot_params: dict):
    """
    Generates a Plotly figure.
    plot_params is a dictionary where keys are plot arguments (x, y, color, etc.)
    and values are column names from the df or specific values.
    """
    if df is None or df.empty:
        return go.Figure(layout={"title": "No data to display or data is empty."})
    if not plot_type_value:
        return go.Figure(layout={"title": "Please select a plot type."})

    # Clean params: remove None values and ensure essential args like x or names are present
    cleaned_params = {k: v for k, v in plot_params.items() if v is not None and v != ""}
    
    # Convert numeric columns if they are of object type due to mixed data or initial load
    # This is important for many plot types
    for arg, col_name in cleaned_params.items():
        if isinstance(col_name, str) and col_name in df.columns:
            # Heuristic: If arg is typically numeric (e.g. x, y, z, size, values for non-pie)
            typical_numeric_args = ['x', 'y', 'z', 'size', 'values', 'nbins', 'nbinsx', 'nbinsy']
            typical_categorical_args = ['names', 'parents', 'color', 'symbol', 'line_group', 'facet_row', 'facet_col']

            if arg in typical_numeric_args and df[col_name].dtype == 'object':
                try:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    print(f"Plot Util: Coerced column '{col_name}' for arg '{arg}' to numeric.")
                except Exception as e:
                    print(f"Plot Util: Failed to coerce column '{col_name}' to numeric for arg '{arg}': {e}")
            
            # For pie charts, 'values' should be numeric. 'names' is categorical.
            if plot_type_value == 'pie' and arg == 'values' and df[col_name].dtype == 'object':
                 try:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                 except Exception: pass


    # Basic check for required arguments based on plot type
    required_args_missing = False
    if plot_type_value in ["scatter", "line", "bar", "histogram", "box", "violin", "density_heatmap"]:
        if not cleaned_params.get("x") and not cleaned_params.get("y"): # Hist can be just x or y
            if plot_type_value == "histogram" and (cleaned_params.get("x") or cleaned_params.get("y")):
                pass # ok for histogram
            else:
                required_args_missing = True
    elif plot_type_value in ["pie", "sunburst", "treemap"]:
        if not cleaned_params.get("names") and not cleaned_params.get("path"):
            required_args_missing = True
        if plot_type_value != "pie" and not cleaned_params.get("parents") and not cleaned_params.get("path"):
            # Sunburst/Treemap need parents if not using path
            pass # path is an alternative

    if required_args_missing:
         return go.Figure(layout={"title": f"Missing required arguments (e.g., X, Y, or Names) for {plot_type_value}."})

    try:
        plot_function = getattr(px, plot_type_value)
        
        # Some args are specific values, not column names (e.g. nbins, orientation)
        # We need to ensure these are passed correctly if they are not column names.
        final_plot_args = {}
        for k, v in cleaned_params.items():
            if isinstance(v, str) and v in df.columns:
                final_plot_args[k] = df[v] # Pass series directly
            else: # It's a parameter value (e.g., nbins=10, orientation='h')
                # Try to convert to int/float if appropriate for known args
                if k in ['nbins', 'nbinsx', 'nbinsy'] and v:
                    try: final_plot_args[k] = int(v)
                    except ValueError: final_plot_args[k] = v # Keep as string if not int
                elif k in ['trendline_options'] and isinstance(v, str): # e.g. for OLS trendline
                    try: final_plot_args[k] = eval(v) # Be careful with eval
                    except: final_plot_args[k] = v
                else:
                    final_plot_args[k] = v
        
        # Add dataframe as the first argument
        fig = plot_function(df, **final_plot_args)
        
        fig.update_layout(
            title_text=cleaned_params.get("title", f"{plot_type_value.replace('_', ' ').title()}"),
            margin=dict(l=20, r=20, t=40, b=20), # Compact margins
            # template="plotly_white" # Clean template
        )
        return fig
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure(layout={"title": f"Error generating {plot_type_value}: {str(e)}"})