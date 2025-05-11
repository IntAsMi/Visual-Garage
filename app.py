import dash
from dash import dcc, html, dash_table, CeleryManager, DiskcacheManager
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
from flask_caching import Cache
import pandas as pd
import base64
import io
import os
import uuid
import time
import json
from pathlib import Path
from plotly import graph_objects as go

# Import your custom modules
from file_handler import FileLoader, get_sheet_names_from_excel_bytes
from stats_generator import get_column_statistics
from plot_utils import PLOT_TYPES, get_plot_arguments, estimate_plot_time, create_plotly_figure
from pandas_transformer import PandasTransformer

# --- App Initialization ---
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                suppress_callback_exceptions=True, # Essential for dynamic UI
                # For background callbacks if you scale to long processes
                # background_callback_manager=DiskcacheManager(cache_dir="./cache") 
                )
server = app.server

# --- Caching Configuration ---
# Using Flask-Caching for server-side storage of DataFrames
# This simple setup uses the filesystem. For production, use Redis or Memcached.
CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_THRESHOLD': 200 # Max number of items in cache
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

# --- UI Helper Functions ---
def build_lhs_controls(columns_options=None):
    if columns_options is None:
        columns_options = []

    plot_type_dropdown = dbc.Card([
        dbc.CardHeader("1. Select Plot Type"),
        dbc.CardBody([
            dcc.Dropdown(
                id='plot-type-selector',
                options=[{'label': k, 'value': v} for k, v in PLOT_TYPES.items()],
                placeholder="Choose a plot..."
            )
        ])
    ], className="mb-3")

    plot_config_area = dbc.Card([
        dbc.CardHeader("2. Configure Plot Arguments"),
        dbc.CardBody(id='dynamic-plot-options-container', children=[
            html.P("Select a plot type to see its options.")
        ])
    ], className="mb-3")
    
    transformation_area = dbc.Card([
        dbc.CardHeader("3. Data Transformations (Optional)"),
        dbc.CardBody([
            html.Div(id='transformations-list-container', children=[]), # Where transformation steps will be added
            dbc.Button("Add Transformation Step", id='add-transformation-button', n_clicks=0, className="mt-2", size="sm"),
            dbc.Button("Apply Transformations", id='apply-transformations-button', n_clicks=0, color="primary", className="mt-2 ms-2", size="sm", disabled=True)
        ])
    ], className="mb-3")

    return html.Div([
        plot_type_dropdown,
        plot_config_area,
        transformation_area,
        dbc.Button(children=[html.I(className="fas fa-chart-line me-2"), "Generate/Update Plot"], id='generate-plot-button', n_clicks=0, color="success", className="w-100 mt-3", disabled=True)
    ])

def build_plot_arg_dropdown(arg_name, columns_options, multi=False):
    # Common arguments that might not be columns but direct values
    non_column_args_inputs = {
        "title": dcc.Input(type="text", placeholder="Plot Title"),
        "nbins": dcc.Input(type="number", placeholder="e.g., 20", min=1, step=1),
        "nbinsx": dcc.Input(type="number", placeholder="e.g., 20", min=1, step=1),
        "nbinsy": dcc.Input(type="number", placeholder="e.g., 20", min=1, step=1),
        "orientation": dcc.Dropdown(options=[{'label': o, 'value': o} for o in ['v', 'h']]),
        "barmode": dcc.Dropdown(options=[{'label': o.title(), 'value': o} for o in ['group', 'stack', 'relative', 'overlay']]),
        "histfunc": dcc.Dropdown(options=[{'label': o.title(), 'value': o} for o in ['count', 'sum', 'avg', 'min', 'max']]),
        "barnorm": dcc.Dropdown(options=[{'label': 'None', 'value': ''}] + [{'label': o.title(), 'value': o} for o in ['fraction', 'percent']]),
        "points": dcc.Dropdown(options=[{'label': str(o).title(), 'value': o} for o in ['all', 'outliers', 'suspectedoutliers', False]]),
        "trendline": dcc.Dropdown(options=[{'label': 'None', 'value': ''}] + [{'label': o.upper(), 'value': o} for o in ['ols', 'lowess', 'expanding', 'rolling']]),
        # Add more as needed
    }
    if arg_name in non_column_args_inputs:
        # For specific args, provide a direct input instead of a column selector
        return html.Div([
            dbc.Label(f"{arg_name.replace('_', ' ').title()}:", html_for={'type': 'plot-arg-input', 'index': arg_name}),
            non_column_args_inputs[arg_name]
        ], className="mb-2")

    return html.Div([
        dbc.Label(f"{arg_name.replace('_', ' ').title()}:", html_for={'type': 'plot-arg-input', 'index': arg_name}),
        dcc.Dropdown(
            options=columns_options,
            placeholder=f"Select column(s) for {arg_name}",
            multi=multi,
        )
    ], className="mb-2", id={'type': 'plot-arg-input', 'index': arg_name}) # Give ID to the Div container for easy access

# --- App Layout ---
app.layout = dbc.Container(fluid=True, children=[
    # Stores
    dcc.Store(id='session-df-id'),      # Stores unique ID for cached original DataFrame
    dcc.Store(id='transformed-df-id'),  # Stores unique ID for cached transformed DataFrame
    dcc.Store(id='current-columns-list'), # Stores list of current columns for dropdowns
    dcc.Store(id='file-info-store'),      # Stores file metadata

    # Header
    dbc.Row(dbc.Col(html.H2("Plotly Studio - CSV Visualizer", className="text-center my-3"))),

    # Upload Area or Main Content
    html.Div(id='upload-section', children=[
        dbc.Row(dbc.Col(dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select a CSV/Excel File')]),
            style={
                'width': '100%', 'height': '100px', 'lineHeight': '100px',
                'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '20px 0'
            },
            multiple=False # Allow one file at a time
        ), width=12)),
        dbc.Row(dbc.Col(html.Div(id='output-file-info', className="mt-2 text-center small"))),
        # Excel sheet selector (hidden initially)
        dbc.Row(dbc.Col(id='excel-sheet-selector-container', className="mt-2", width={"size": 6, "offset": 3})) 
    ]),
    
    html.Div(id='main-app-content', style={'display': 'none'}, children=[
        dbc.Row([
            # LHS: Controls
            dbc.Col(md=3, children=[
                html.Div(id='lhs-controls-container', className="lhs-panel")
            ], className="bg-light p-3"),

            # RHS: Data Preview & Plot
            dbc.Col(md=9, children=[
                dbc.Row([
                    dbc.Col(html.H5("Data Preview (First 100 Rows)"), width=8),
                    dbc.Col(html.Div(id="row-col-info", className="text-end small"), width=4)
                ]),
                html.Div(id='data-preview-container', className="rhs-data-preview", children=[
                    dash_table.DataTable(
                        id='data-preview-table',
                        page_size=10, # Show fewer rows for faster rendering in preview
                        style_table={'overflowX': 'auto', 'minWidth': '100%'},
                        style_header={'fontWeight': 'bold'},
                        sort_action='native',
                        filter_action='native',
                        column_selectable="single" # For stats
                    )
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col(html.Div(id='column-stats-display', className="small", style={"maxHeight": "200px", "overflowY": "auto"})),
                ]),
                html.Hr(),
                dbc.Row([
                     dbc.Col(html.H5("Plot Preview"), width=10),
                     dbc.Col(html.Div(id='plot-time-estimation-warning', className="small text-warning text-end"),width=2)
                ]),
                dcc.Loading(id="loading-plot", type="default", children=[
                    dcc.Graph(id='plotly-graph-preview', style={'height': '45vh'})
                ], className="plot-preview-area")
            ])
        ])
    ])
])

# --- Callbacks ---

@app.callback(
    [Output('excel-sheet-selector-container', 'children'),
     Output('output-file-info', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def display_excel_sheet_selector(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update

    file_ext = Path(filename).suffix.lower()
    if file_ext in ['.xlsx', '.xlsm', '.xls', '.xlsb']:
        try:
            content_type, content_string = contents.split(',')
            decoded_bytes = base64.b64decode(content_string)
            sheet_names = get_sheet_names_from_excel_bytes(decoded_bytes, file_ext)
            
            if not sheet_names:
                return None, dbc.Alert("Could not read sheet names from this Excel file.", color="danger")

            selector = html.Div([
                dbc.Label("Select Excel Sheet:"),
                dcc.Dropdown(id='excel-sheet-name-selector', 
                             options=[{'label': s, 'value': s} for s in sheet_names], 
                             value=sheet_names[0] if sheet_names else None),
                dbc.Button("Load Selected Sheet", id='load-excel-button', n_clicks=0, className="mt-2", color="primary")
            ])
            return selector, f"File: {filename}. Found {len(sheet_names)} sheet(s). Please select one to load."
        except Exception as e:
            return None, dbc.Alert(f"Error processing Excel file {filename}: {e}", color="danger")
    return None, "" # No selector for non-excel or if already loaded

@app.callback(
    [Output('session-df-id', 'data'),
     Output('transformed-df-id', 'data'), # Reset transformed
     Output('file-info-store', 'data'),
     Output('main-app-content', 'style'),
     Output('upload-section', 'style'),
     Output('lhs-controls-container', 'children'),
     Output('current-columns-list', 'data'),
     Output('output-file-info', 'children', allow_duplicate=True),
     Output('generate-plot-button', 'disabled'),
     Output('apply-transformations-button', 'disabled'),
     Output('excel-sheet-selector-container', 'children', allow_duplicate=True)], # Clear sheet selector after load
    [Input('upload-data', 'contents'), # For CSVs
     Input('load-excel-button', 'n_clicks')], # For Excels
    [State('upload-data', 'filename'),
     State('upload-data', 'contents'), # Need contents again for excel if triggered by button
     State('excel-sheet-name-selector', 'value')],
    prevent_initial_call=True
)
@cache.memoize(timeout=3600) # Cache result for 1 hour based on inputs
def handle_file_upload_and_load(csv_contents, excel_n_clicks, filename, excel_contents_state, selected_sheet):
    """Handles both CSV upload directly and Excel load via button click."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    file_content_to_process = None
    sheet_name_to_load = None

    if trigger_id == 'upload-data' and Path(filename).suffix.lower() == '.csv':
        file_content_to_process = csv_contents
    elif trigger_id == 'load-excel-button' and excel_contents_state and selected_sheet:
        file_content_to_process = excel_contents_state # Use the state for excel contents
        sheet_name_to_load = selected_sheet
    elif trigger_id == 'upload-data' and Path(filename).suffix.lower() in ['.xlsx', '.xlsm', '.xls', '.xlsb']:
        # Excel file dropped, but sheet not selected yet. Don't load, wait for button.
        return dash.no_update * (len(ctx.outputs_list[0])+1) # Send no_update for all outputs

    if file_content_to_process is None or filename is None:
        return dash.no_update * (len(ctx.outputs_list[0])+1)

    start_load_time = time.time()
    loader = FileLoader(file_content_string=file_content_to_process, filename=filename, sheet_name=sheet_name_to_load)
    df, file_info = loader.run()
    
    file_info['user_load_duration'] = time.time() - start_load_time

    if df.empty or 'error' in file_info:
        err_msg = file_info.get('error', 'Failed to load data or data is empty.')
        return (None, None, file_info, {'display': 'none'}, {'display': 'block'}, 
                build_lhs_controls(), [], dbc.Alert(f"Error: {err_msg}", color="danger"), 
                True, True, None if trigger_id == 'load-excel-button' else dash.no_update)

    # Successfully loaded
    df_id = str(uuid.uuid4())
    cache.set(df_id, df) # Store in Flask-Caching

    columns = [{'label': str(col), 'value': str(col)} for col in df.columns]
    
    info_text = (f"Loaded: {file_info.get('filename', 'N/A')} | "
                 f"{file_info.get('rows', 0):,} Rows, {file_info.get('columns', 0):,} Cols | "
                 f"Size: {file_info.get('size_mb', 0):.2f}MB | "
                 f"Load Time: {file_info.get('load_time_seconds', 0):.2f}s")
    if 'sheet_name' in file_info: info_text += f" | Sheet: {file_info['sheet_name']}"
    if 'read_method' in file_info: info_text += f" | Method: {file_info['read_method']}"

    return (df_id, df_id, file_info, {'display': 'block'}, {'display': 'none'}, 
            build_lhs_controls(columns), columns, html.Div([
                dbc.Alert("File loaded successfully!", color="success", duration=4000),
                html.P(info_text, className="small")
            ]), 
            False, False, None if trigger_id == 'load-excel-button' else dash.no_update) # Clear sheet selector


@app.callback(
    Output('data-preview-table', 'columns'),
    Output('data-preview-table', 'data'),
    Output('row-col-info', 'children'),
    Input('transformed-df-id', 'data'), # Use transformed df for preview
    State('current-columns-list', 'data')
)
def update_data_preview(df_id, columns_list_for_header):
    if not df_id:
        return [], [], ""
    
    try:
        df = cache.get(df_id)
        if df is None: return [], [], "Error: DataFrame not found in cache."
    except Exception as e:
        return [], [], f"Error retrieving df from cache: {e}"


    preview_df = df.head(100) # Always show head of potentially transformed df
    
    # Use current-columns-list for headers if available and matches df, else derive from preview_df
    # This helps if columns were renamed/dropped.
    table_cols = []
    if columns_list_for_header:
        # Filter to ensure only columns present in the current df are used for the table
        valid_cols_for_table = [col_dict['value'] for col_dict in columns_list_for_header if col_dict['value'] in preview_df.columns]
        table_cols = [{'label': c, 'name': c, 'id': c} for c in valid_cols_for_table]
    else: # Fallback
        table_cols = [{'label': str(col), 'name': str(col), 'id': str(col)} for col in preview_df.columns]

    data = preview_df.to_dict('records')
    row_col_text = f"{df.shape[0]:,} Rows, {df.shape[1]:,} Columns"
    return table_cols, data, row_col_text


@app.callback(
    Output('column-stats-display', 'children'),
    Input('data-preview-table', 'selected_columns'),
    State('transformed-df-id', 'data')
)
def display_column_stats(selected_columns, df_id):
    if not selected_columns or not df_id:
        return "Select a column header in the preview table to see its statistics."
    
    try:
        df = cache.get(df_id)
        if df is None: return "Error: DataFrame not found in cache for stats."
    except Exception as e:
         return f"Error retrieving df from cache for stats: {e}"
        
    col_name = selected_columns[0]
    if col_name not in df.columns:
        return f"Error: Column '{col_name}' not found in the current DataFrame."
    
    stats_markdown = get_column_statistics(df[col_name])
    return dcc.Markdown(stats_markdown)


@app.callback(
    Output('dynamic-plot-options-container', 'children'),
    Input('plot-type-selector', 'value'),
    State('current-columns-list', 'data')
)
def update_plot_options_ui(plot_type_value, columns_options):
    if not plot_type_value:
        return html.P("Select a plot type to see its options.")
    if not columns_options:
        return html.P("Load data to configure plot options.")

    args_for_plot = get_plot_arguments(plot_type_value)
    # Filter out df itself or other non-user args
    valid_args = [arg for arg in args_for_plot if arg not in ['data_frame', 'self'] and not arg.startswith('_')]

    ui_elements = []
    for arg in valid_args:
        # Determine if argument typically takes multiple columns (e.g., hover_data)
        is_multi = arg in ["hover_data", "custom_data"]
        # Some args might be booleans or specific dropdowns, not column selectors
        # Build specific UI components for such arguments
        ui_elements.append(
            build_plot_arg_dropdown(arg, columns_options, multi=is_multi)
        )
    return ui_elements


@app.callback(
    Output('plot-time-estimation-warning', 'children'),
    Output('plotly-graph-preview', 'figure'),
    Input('generate-plot-button', 'n_clicks'),
    [State('transformed-df-id', 'data'),
     State('plot-type-selector', 'value'),
     State('current-columns-list', 'data')] + # To get all column names
    [State({'type': 'plot-arg-input', 'index': ALL}, 'value'), # Get all plot arg values
     State({'type': 'plot-arg-input', 'index': ALL}, 'id')] # Get all plot arg names (their IDs)
)
def render_plot(n_clicks, df_id, plot_type_value, columns_list, arg_values, arg_ids):
    if n_clicks == 0 or not df_id or not plot_type_value:
        return "", go.Figure(layout={"title": "Configure and click 'Generate Plot'"})

    try:
        df = cache.get(df_id)
        if df is None: return "Error: DataFrame not found.", go.Figure(layout={"title": "Error: Data not found."})
    except Exception as e:
        return f"Error retrieving data: {e}", go.Figure(layout={"title": f"Error: {e}"})

    plot_params = {}
    if arg_ids and arg_values:
        for i, id_dict in enumerate(arg_ids):
            arg_name = id_dict['index'] # The 'index' in the ID pattern is the arg name
            plot_params[arg_name] = arg_values[i]
    
    # Basic estimation
    num_dimensions_for_plot = sum(1 for val in plot_params.values() if isinstance(val, str) and val in df.columns)
    num_dimensions_for_plot = max(1, num_dimensions_for_plot) # At least one dimension
    
    est_time = estimate_plot_time(len(df), num_dimensions_for_plot, plot_type_value)
    warning_msg = f"Est. ~{est_time:.1f}s" if est_time > 3 else "" # Show if > 3s

    fig = create_plotly_figure(df.copy(), plot_type_value, plot_params) # Send a copy to plotting func

    return warning_msg, fig

# --- Callbacks for Transformations ---
TRANSFORMATION_OPTIONS = [
    {'label': 'Convert to Numeric', 'value': 'to_numeric'},
    {'label': 'Convert to Datetime', 'value': 'to_datetime'},
    {'label': 'Convert to String', 'value': 'to_string'},
    {'label': 'Fill Missing Values (NA)', 'value': 'fill_na'},
    {'label': 'Create Column from Formula (Experimental)', 'value': 'create_from_formula'},
    {'label': 'Rename Column', 'value': 'rename_column'},
    {'label': 'Drop Column', 'value': 'drop_column'},
]

def create_transformation_step_ui(step_index, columns_options):
    return dbc.Card(className="mb-2", children=[
        dbc.CardHeader(f"Step {step_index + 1}", className="p-2"),
        dbc.CardBody(className="p-2", children=[
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id={'type': 'transform-op-selector', 'index': step_index},
                    options=TRANSFORMATION_OPTIONS, placeholder="Select Operation"
                ), md=4),
                dbc.Col(dcc.Dropdown(
                    id={'type': 'transform-col-selector', 'index': step_index},
                    options=columns_options, placeholder="Select Source Column"
                ), md=4),
                dbc.Col(dbc.Input(
                    id={'type': 'transform-new-col-input', 'index': step_index},
                    placeholder="New/Target Column Name (optional)"
                ), md=3),
                 dbc.Col(dbc.Button(html.I(className="fas fa-times"), 
                                   id={'type': 'remove-transform-step-button', 'index': step_index}, 
                                   color="danger", size="sm",className="float-end"), md=1)
            ]),
            html.Div(id={'type': 'transform-params-container', 'index': step_index}, className="mt-2")
        ])
    ])

@app.callback(
    Output('transformations-list-container', 'children'),
    Output('apply-transformations-button', 'disabled', allow_duplicate=True),
    Input('add-transformation-button', 'n_clicks'),
    Input({'type': 'remove-transform-step-button', 'index': ALL}, 'n_clicks'),
    State('transformations-list-container', 'children'),
    State('current-columns-list', 'data'),
    prevent_initial_call=True
)
def manage_transformation_steps(add_clicks, remove_clicks_list, current_steps_ui, columns_options):
    ctx = dash.callback_context
    triggered_id_str = ctx.triggered[0]['prop_id']

    if not columns_options: # Don't add steps if no data/columns
        return [], True 

    if "add-transformation-button" in triggered_id_str:
        new_step_index = len(current_steps_ui)
        new_step_ui = create_transformation_step_ui(new_step_index, columns_options)
        current_steps_ui.append(new_step_ui)
        return current_steps_ui, False if current_steps_ui else True

    if "remove-transform-step-button" in triggered_id_str:
        try:
            triggered_id_dict = json.loads(triggered_id_str.split('.')[0]) # prop_id is like '{"index":0,"type":"remove-transform-step-button"}.n_clicks'
            remove_index = triggered_id_dict['index']
            
            # Rebuild UI excluding the removed step and re-indexing subsequent ones
            updated_steps_ui = []
            current_step_data_for_rebuild = [] # If you need to preserve values, this is more complex

            original_index_counter = 0
            for i, step_ui_component in enumerate(current_steps_ui):
                # Assume Card has an ID or identifiable property if parsing existing values is needed.
                # Here, we just reconstruct based on original list excluding the one to remove.
                if i == remove_index:
                    continue 
                
                # Re-create the step with a new index for UI consistency
                # This simplified version just re-adds. A more robust version would preserve user inputs.
                updated_steps_ui.append(create_transformation_step_ui(len(updated_steps_ui), columns_options))

            return updated_steps_ui, False if updated_steps_ui else True
        except Exception as e:
            print(f"Error removing step: {e}")
            # Fallback to just returning current steps if parsing ID fails
            return current_steps_ui, False if current_steps_ui else True


    return current_steps_ui, False if current_steps_ui else True

@app.callback(
    Output({'type': 'transform-params-container', 'index': MATCH}, 'children'),
    Input({'type': 'transform-op-selector', 'index': MATCH}, 'value'),
)
def display_transformation_params(operation):
    if not operation:
        return []
    
    params_ui = []
    if operation == 'to_numeric':
        params_ui.append(dcc.Dropdown(id={'type':'transform-param-errors', 'index':MATCH}, options=[{'label':e.title(), 'value':e} for e in ['coerce', 'raise', 'ignore']], value='coerce', placeholder="Error Handling"))
    elif operation == 'to_datetime':
        params_ui.append(dcc.Dropdown(id={'type':'transform-param-errors', 'index':MATCH}, options=[{'label':e.title(), 'value':e} for e in ['coerce', 'raise', 'ignore']], value='coerce', placeholder="Error Handling"))
        params_ui.append(dbc.Input(id={'type':'transform-param-format', 'index':MATCH}, placeholder="Datetime Format (e.g., %Y-%m-%d), optional", className="mt-1"))
    elif operation == 'fill_na':
        params_ui.append(dbc.Input(id={'type':'transform-param-value', 'index':MATCH}, placeholder="Value to fill NA with (e.g., 0 or 'Unknown')", className="mt-1"))
    elif operation == 'create_from_formula':
        params_ui.append(dbc.Textarea(id={'type':'transform-param-formula', 'index':MATCH}, placeholder="Enter formula (e.g., df['col_A'] * 2 / df['col_B'])\nWARNING: Experimental & limited safety.", rows=2, className="mt-1"))
        params_ui.append(dbc.Alert("Use column names within df['col_name'] or directly if simple. Complex functions not supported by basic df.eval. Exercise caution.", color="warning", className="small mt-1"))
    elif operation == 'rename_column':
        params_ui.append(dbc.Input(id={'type':'transform-param-new_name_parameter', 'index':MATCH}, placeholder="Enter new column name", className="mt-1", required=True))

    return dbc.Row([dbc.Col(p) for p in params_ui], className="g-2") # g-2 for spacing

@app.callback(
    [Output('transformed-df-id', 'data', allow_duplicate=True),
     Output('current-columns-list', 'data', allow_duplicate=True),
     Output('output-file-info', 'children', allow_duplicate=True), # To show messages
     Output('lhs-controls-container', 'children', allow_duplicate=True)], # Rebuild LHS if cols change
    Input('apply-transformations-button', 'n_clicks'),
    [State('session-df-id', 'data'), # Start from original or last transformed? For now, start from original each time.
     State('transformations-list-container', 'children'), # To know how many steps
     # Selectors for ALL transformation steps' values
     State({'type': 'transform-op-selector', 'index': ALL}, 'value'),
     State({'type': 'transform-col-selector', 'index': ALL}, 'value'),
     State({'type': 'transform-new-col-input', 'index': ALL}, 'value'),
     # Param selectors - this will get complex if params are very different.
     # A more robust way is to iterate through children of 'transform-params-container' for each step.
     # Simplified: Assume we can get them if they exist.
     State({'type': 'transform-param-errors', 'index': ALL}, 'value'),
     State({'type': 'transform-param-format', 'index': ALL}, 'value'),
     State({'type': 'transform-param-value', 'index': ALL}, 'value'),
     State({'type': 'transform-param-formula', 'index': ALL}, 'value'),
     State({'type': 'transform-param-new_name_parameter', 'index': ALL}, 'value'),
     State('current-columns-list', 'data') # Original columns for LHS rebuild
     ],
    prevent_initial_call=True
)
def apply_all_transformations(n_clicks, source_df_id, num_steps_ui, 
                              ops, src_cols, new_cols_names,
                              param_errors_list, param_format_list, param_value_list, param_formula_list, param_rename_new_name_list,
                              original_cols_for_lhs_rebuild):
    if n_clicks == 0 or not source_df_id or not ops:
        return dash.no_update, dash.no_update, dbc.Alert("No transformations to apply.", color="info", duration=3000), dash.no_update

    try:
        df_current = cache.get(source_df_id)
        if df_current is None:
            return source_df_id, original_cols_for_lhs_rebuild, dbc.Alert("Error: Source DataFrame not found.", color="danger"), build_lhs_controls(original_cols_for_lhs_rebuild) # Rebuild with original columns
    except Exception as e:
        return source_df_id, original_cols_for_lhs_rebuild, dbc.Alert(f"Error getting df: {e}", color="danger"), build_lhs_controls(original_cols_for_lhs_rebuild)


    transformer = PandasTransformer(df_current.copy()) # Always work on a fresh copy for a full transformation sequence
    messages = []

    # Map available param lists to their operations for easier lookup
    # This is brittle if param component IDs change or new ones are added without updating here.
    param_map = {
        'to_numeric': {'errors': param_errors_list},
        'to_datetime': {'errors': param_errors_list, 'format': param_format_list},
        'fill_na': {'value': param_value_list},
        'create_from_formula': {'formula_string': param_formula_list},
        'rename_column': {'new_name_parameter': param_rename_new_name_list}
    }
    
    # Track current index for each param list as we iterate through ops
    param_indices = {k: 0 for k in ['errors', 'format', 'value', 'formula', 'new_name_parameter']}

    for i in range(len(ops)):
        operation = ops[i]
        source_col = src_cols[i]
        new_col_name = new_cols_names[i] if new_cols_names[i] else source_col

        if not operation or (not source_col and operation not in ['drop_column', 'rename_column']): # Drop/Rename might only need source
             if operation in ['drop_column', 'rename_column'] and not source_col:
                messages.append(f"Step {i+1}: Skipped - Source column missing for {operation}.")
                continue
             elif not operation:
                messages.append(f"Step {i+1}: Skipped - Operation not selected.")
                continue
        
        current_params = {}
        if operation in param_map:
            for param_key, param_list_ref_key in param_map[operation].items(): # param_list_ref_key is 'errors', 'format' etc.
                param_list_values = locals().get(f"param_{param_list_ref_key}_list") # e.g., param_errors_list
                if param_list_values and param_indices[param_list_ref_key] < len(param_list_values):
                    current_params[param_key] = param_list_values[param_indices[param_list_ref_key]]
                    param_indices[param_list_ref_key] += 1
                # else: current_params[param_key] = None # or some default

        df_transformed, msg = transformer.apply_transformation(source_col, operation, new_col_name, current_params)
        messages.append(f"Step {i+1} ({operation} on '{source_col}'): {msg}")
        if "Error" in msg: # If a step fails, stop further transformations for this run
            final_df_id = str(uuid.uuid4()) # Cache the partially transformed DF
            cache.set(final_df_id, df_transformed)
            updated_cols = [{'label': str(col), 'value': str(col)} for col in df_transformed.columns]
            return final_df_id, updated_cols, html.Div([dbc.Alert(m, color="info" if "success" in m.lower() else "warning" if "skip" in m.lower() else "danger") for m in messages]), build_lhs_controls(updated_cols)


    final_df = transformer.get_dataframe()
    transformed_df_id_new = str(uuid.uuid4())
    cache.set(transformed_df_id_new, final_df)

    updated_columns_list = [{'label': str(col), 'value': str(col)} for col in final_df.columns]
    
    # Rebuild LHS controls with new column list
    new_lhs_controls = build_lhs_controls(updated_columns_list)

    return transformed_df_id_new, updated_columns_list, html.Div([dbc.Alert(m, color="success" if "success" in m.lower() else "warning" if "skip" in m.lower() else "danger", className="small") for m in messages]), new_lhs_controls


# --- Run the App ---
if __name__ == '__main__':
    if not os.path.exists('cache-directory'):
        os.makedirs('cache-directory')
    app.run(debug=True, host='0.0.0.0') # host='0.0.0.0' for Docker or network access