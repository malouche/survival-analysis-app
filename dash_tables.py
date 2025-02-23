from dash import dash_table
import pandas as pd

def format_number(value):
    """Format numbers to 3 decimal places for floats, no decimals for integers."""
    if isinstance(value, (int, np.integer)):
        return f"{value:d}"
    elif isinstance(value, float):
        return f"{value:.3f}"
    return value

def create_results_table(df):
    """Create a Dash table for the results."""
    # Format the numeric columns
    formatted_df = df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(format_number)
    
    return dash_table.DataTable(
        data=formatted_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in formatted_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'minWidth': '100px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'border': '1px solid black'
        },
        style_data={
            'border': '1px solid grey'
        }
    )

def create_input_table(df):
    """Create a Dash table for the input data."""
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        style_table={
            'height': '300px',
            'overflowY': 'auto',
            'overflowX': 'auto',
            'margin': '0 auto',  # Center the table
            'width': '50%'  # Adjust width as needed
        },
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'minWidth': '100px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'border': '1px solid black'
        },
        style_data={
            'border': '1px solid grey'
        }
    )
