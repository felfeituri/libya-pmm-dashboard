"""
Libya PMM - Interactive Dashboard
Plotly Dash application for Price Market Monitoring visualization and analysis

Features:
- Interactive maps showing municipality-level MEB
- Time series charts for national and regional trends (gap-linked)
- Municipality-level MEB heatmap
- Municipality rankings and comparisons
- Commodity price table with MoM%
- Data export capabilities

Usage:
    python pmm_dashboard.py
    
Then open browser to: http://127.0.0.1:8051
"""
import os
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from pathlib import Path

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('PMM_DB_HOST'),
    'port': int(os.getenv('PMM_DB_PORT', '5432')),
    'database': os.getenv('PMM_DB_NAME'),
    'user': os.getenv('PMM_DB_USER'),
    'password': os.getenv('PMM_DB_PASSWORD'),
}

def get_engine():
    """Create SQLAlchemy engine"""
    conn_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(conn_string)

# ============================================================================
# WFP BRANDING COLORS
# ============================================================================

COLORS = {
    'primary': '#0A6EB4',      # WFP Blue
    'secondary': '#F2F2F2',    # Light gray background
    'text': '#595959',         # Dark gray text
    'east': '#9BC5AC',         # Green
    'west': '#FBB189',         # Orange
    'south': '#EA8C8C',        # Red
    'national': '#73B4E0',     # Light blue
    'success': '#00B050',      # Green for positive
    'warning': '#C00000',      # Red for negative
    'background': '#FFFFFF',   # White
    'border': '#D9D9D9',       # Light border
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_latest_date():
    """Get the most recent date with data"""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(date) FROM national_meb WHERE full_meb IS NOT NULL AND full_meb > 0")
        )
        return result.scalar()

def get_date_range():
    """Get available date range"""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT MIN(date) as min_date, MAX(date) as max_date 
            FROM national_meb 
            WHERE full_meb IS NOT NULL AND full_meb > 0
        """))
        row = result.fetchone()
        return row[0], row[1]

def get_national_overview(target_date, region_scope='National'):
    """
    Get MEB overview with MoM changes.
    region_scope: 'National', 'East', 'West', 'South'
    """
    engine = get_engine()

    target_dt = pd.to_datetime(target_date)
    # Previous calendar month
    if target_dt.month == 1:
        prev_month = target_dt.replace(year=target_dt.year - 1, month=12, day=1)
    else:
        prev_month = target_dt.replace(month=target_dt.month - 1, day=1)

    if region_scope == 'National':
        current_query = text("""
            SELECT food_meb, nfi_meb, full_meb
            FROM national_meb
            WHERE date = :date
        """)
        prev_query = text("""
            SELECT food_meb, nfi_meb, full_meb
            FROM national_meb
            WHERE date = :date
        """)
        cur_params = {'date': target_date}
        prev_params = {'date': prev_month}
    else:
        # Regional KPIs from regional_meb
        current_query = text("""
            SELECT food_meb, nfi_meb, full_meb
            FROM regional_meb
            WHERE date = :date AND region = :region
        """)
        prev_query = text("""
            SELECT food_meb, nfi_meb, full_meb
            FROM regional_meb
            WHERE date = :date AND region = :region
        """)
        cur_params = {'date': target_date, 'region': region_scope}
        prev_params = {'date': prev_month, 'region': region_scope}
    
    with engine.connect() as conn:
        current = conn.execute(current_query, cur_params).fetchone()
        previous = conn.execute(prev_query, prev_params).fetchone()
    
    if not current:
        return None
    
    result = {
        'food_meb': float(current[0]) if current[0] else 0,
        'nfi_meb': float(current[1]) if current[1] else 0,
        'full_meb': float(current[2]) if current[2] else 0,
    }
    
    if previous:
        result['food_meb_mom'] = (
            (result['food_meb'] - float(previous[0])) / float(previous[0]) * 100
        ) if previous[0] else 0
        result['nfi_meb_mom'] = (
            (result['nfi_meb'] - float(previous[1])) / float(previous[1]) * 100
        ) if previous[1] else 0
        result['full_meb_mom'] = (
            (result['full_meb'] - float(previous[2])) / float(previous[2]) * 100
        ) if previous[2] else 0
    else:
        result['food_meb_mom'] = 0
        result['nfi_meb_mom'] = 0
        result['full_meb_mom'] = 0
    
    return result

def get_municipality_data(target_date):
    """Get all municipality data for given date"""
    engine = get_engine()
    
    query = text("""
        SELECT 
            m.municipality,
            m.food_meb,
            m.nfi_meb,
            m.full_meb,
            l.x as longitude,
            l.y as latitude,
            l.adm1_en as region
        FROM municipality_meb m
        JOIN locations l ON m.adm2_pcode = l.adm2_pcode
        WHERE m.date = :date
        AND m.full_meb IS NOT NULL
        AND m.full_meb > 0
        ORDER BY m.full_meb DESC
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {'date': target_date})
        df = pd.DataFrame(
            result.fetchall(), 
            columns=['municipality', 'food_meb', 'nfi_meb', 'full_meb', 
                     'longitude', 'latitude', 'region']
        )
    
    # Convert to float
    for col in ['food_meb', 'nfi_meb', 'full_meb', 'longitude', 'latitude']:
        df[col] = df[col].astype(float)
    
    return df

def get_trend_data(meb_type='full_meb', months=12):
    """Get national and regional trend data"""
    engine = get_engine()
    
    # National trends
    nat_query = text(f"""
        SELECT date, {meb_type}
        FROM national_meb
        WHERE {meb_type} IS NOT NULL AND {meb_type} > 0
        ORDER BY date DESC
        LIMIT :months
    """)
    
    # Regional trends
    reg_query = text(f"""
        SELECT date, region, {meb_type}
        FROM regional_meb
        WHERE {meb_type} IS NOT NULL AND {meb_type} > 0
        ORDER BY region, date DESC
    """)
    
    with engine.connect() as conn:
        nat_df = pd.read_sql(
            nat_query, conn, 
            params={'months': months if months != 999 else 1000}
        )
        nat_df = nat_df.sort_values('date')
        nat_df[meb_type] = nat_df[meb_type].astype(float)
        
        reg_df = pd.read_sql(reg_query, conn)
        reg_df = reg_df.sort_values(['region', 'date'])
        reg_df[meb_type] = reg_df[meb_type].astype(float)
        
        # Limit each region to last N months (if not "all time")
        if months != 999:
            reg_df = reg_df.groupby('region').tail(months).reset_index(drop=True)
    
    return nat_df, reg_df

def get_commodity_data(target_date):
    """Get commodity price data for target date"""
    engine = get_engine()
    
    query = text("""
        SELECT 
            product_name,
            category,
            admin_name as region,
            average_price
        FROM products
        WHERE date = :date
        ORDER BY category, product_name
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {'date': target_date})
        df = pd.DataFrame(
            result.fetchall(), 
            columns=['product_name', 'category', 'region', 'average_price']
        )
    
    df['average_price'] = df['average_price'].astype(float)
    return df

def get_municipality_heatmap_data(meb_type='full_meb', scope_region='National'):
    """
    Return MEB values by municipality and month for the heatmap,
    filtered by scope (National / East / West / South) and MEB type.

    We do NOT do any time-window trimming here; that is handled in the
    callback so we can always build a full monthly range and show gaps.
    """
    engine = get_engine()
    base_sql = """
        SELECT
            m.date,
            m.municipality,
            l.adm1_en AS region,
            m.food_meb,
            m.nfi_meb,
            m.full_meb
        FROM municipality_meb m
        JOIN locations l ON m.adm2_pcode = l.adm2_pcode
        WHERE m.full_meb IS NOT NULL
          AND m.full_meb > 0
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(base_sql), conn)

    if df.empty:
        return df

    # Filter by scope (National vs region)
    if scope_region != 'National':
        df = df[df['region'] == scope_region]

    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Choose MEB type
    value_col = meb_type  # 'full_meb', 'food_meb', or 'nfi_meb'
    df['meb_value'] = df[value_col].astype(float)

    return df[['date', 'municipality', 'region', 'meb_value']]


# ============================================================================
# GAP-LINKED TREND HELPER
# ============================================================================

def build_gap_segments(series):
    """
    Given a monthly Series with a DateTimeIndex and possible NaNs,
    return:
      - segments: list of (x_list, y_list) for real data segments
      - bridges:  list of (x1, y1, x2, y2) for faded connectors across gaps
      - gap_ranges: list of (gap_start_date, gap_end_date) where data is missing
    """
    series = series.sort_index()
    dates = list(series.index)
    vals = series.values.astype(float)

    n = len(dates)
    if n == 0:
        return [], [], []

    valid = ~np.isnan(vals)

    segments = []
    bridges = []
    gap_ranges = []

    # Real line segments
    i = 0
    while i < n:
        if valid[i]:
            start = i
            while i + 1 < n and valid[i + 1]:
                i += 1
            end = i
            segments.append((dates[start:end + 1], vals[start:end + 1]))
            i += 1
        else:
            i += 1

    # Bridges + gap ranges
    i = 0
    while i < n:
        if valid[i]:
            j = i + 1
            if j < n and not valid[j]:
                # start of a gap
                gap_start = j
                while j < n and not valid[j]:
                    j += 1
                gap_end = j - 1
                if j < n and valid[j]:
                    # we have valid before and after the gap
                    bridges.append((dates[i], vals[i], dates[j], vals[j]))
                    gap_ranges.append((dates[gap_start], dates[gap_end]))
                i = j
            else:
                i += 1
        else:
            i += 1

    return segments, bridges, gap_ranges

# ============================================================================
# INITIALIZE DASH APP
# ============================================================================

app = dash.Dash(
    __name__,
    title='Libya PMM Dashboard',
    suppress_callback_exceptions=True
)

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Img(
                src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzBBNkVCNCIvPgogIDx0ZXh0IHg9IjUwIiB5PSI1MCIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjQwIiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZG9taW5hbnQtYmFzZWxpbmU9Im1pZGRsZSI+V0ZQPC90ZXh0Pgo8L3N2Zz4=',
                style={'height': '60px', 'marginRight': '20px'}
            ),
            html.Div([
                html.H1(
                    'Libya Price Market Monitoring', 
                    style={'margin': '0', 'color': COLORS['text'], 'fontSize': '32px'}
                ),
                html.P(
                    'Interactive Dashboard - Minimum Expenditure Basket Analysis',
                    style={'margin': '5px 0 0 0', 'color': COLORS['text'], 'fontSize': '16px'}
                ),
            ]),
        ], style={'display': 'flex', 'alignItems': 'center'}),
    ], style={
        'backgroundColor': COLORS['background'],
        'padding': '20px 40px',
        'borderBottom': f'3px solid {COLORS["primary"]}',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Date Selector
    html.Div([
        html.Label(
            'Select Month:', 
            style={'fontWeight': 'bold', 'marginRight': '10px', 'color': COLORS['text']}
        ),
        dcc.Dropdown(
            id='date-selector',
            style={'width': '300px'}
        ),
    ], style={
        'padding': '20px 40px',
        'backgroundColor': COLORS['secondary']
    }),
    
    # Main Content
    html.Div([
        # Region filter + Overview Cards
        html.Div([
            html.Div([
                html.Label(
                    'Scope:', 
                    style={
                        'fontWeight': 'bold',
                        'marginRight': '10px',
                        'color': COLORS['text']
                    }
                ),
                dcc.RadioItems(
                    id='overview-region-filter',
                    options=[
                        {'label': ' National', 'value': 'National'},
                        {'label': ' East', 'value': 'East'},
                        {'label': ' West', 'value': 'West'},
                        {'label': ' South', 'value': 'South'},
                    ],
                    value='National',
                    inline=True,
                ),
            ], style={'marginBottom': '10px'}),
            
            html.Div(id='overview-cards', style={'marginBottom': '30px'}),
        ]),
        
        # Tabs
        dcc.Tabs(
            id='main-tabs', 
            value='geographic', 
            children=[
                # Overview Tab
                dcc.Tab(label='Overview', value='geographic', children=[
                    html.Div([
                        html.H2('Municipality MEB Map', style={'color': COLORS['text']}),
                        dcc.RadioItems(
                            id='map-meb-type',
                            options=[
                                {'label': ' Full MEB', 'value': 'full_meb'},
                                {'label': ' Food MEB', 'value': 'food_meb'},
                                {'label': ' NFI MEB', 'value': 'nfi_meb'},
                            ],
                            value='full_meb',
                            inline=True,
                            style={'marginBottom': '20px'}
                        ),
                        dcc.Graph(id='municipality-map', style={'height': '600px'}),
                    ], style={'padding': '20px'})
                ]),
                
                # Trends Tab
                dcc.Tab(label='Time Trends', value='trends', children=[
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label(
                                    'MEB Type:', 
                                    style={'fontWeight': 'bold', 'marginRight': '10px'}
                                ),
                                dcc.RadioItems(
                                    id='trend-meb-type',
                                    options=[
                                        {'label': ' Full MEB', 'value': 'full_meb'},
                                        {'label': ' Food MEB', 'value': 'food_meb'},
                                        {'label': ' NFI MEB', 'value': 'nfi_meb'},
                                    ],
                                    value='full_meb',
                                    inline=True,
                                ),
                            ], style={'marginBottom': '10px'}),
                            html.Div([
                                html.Label(
                                    'Time Period:', 
                                    style={'fontWeight': 'bold', 'marginRight': '10px'}
                                ),
                                dcc.RadioItems(
                                    id='trend-months',
                                    options=[
                                        {'label': ' All Time', 'value': 999},
                                        {'label': ' 12 Months', 'value': 12},
                                        {'label': ' 6 Months', 'value': 6},   
                                    ],
                                    value=999,
                                    inline=True,
                                ),
                            ], style={'marginBottom': '20px'}),
                        ]),
                    html.H2(
                        'MEB Trends Over Time',
                        style={'color': COLORS['text'], 'marginTop': '0', 'marginBottom': '10px'}
                    ),                        
                        dcc.Graph(id='trend-chart', style={'height': '500px'}),

                        # Footnote under the chart
                        html.Div(
                            id='trend-footnote',
                            style={
                                'fontSize': '11px',
                                'color': COLORS['text'],
                                'marginTop': '6px',
                                'fontStyle': 'italic'
                            }
                        ),

                        # Municipality Heatmap section
                        html.H3(
                            'Municipality MEB Heatmap', 
                            style={'color': COLORS['text'], 'marginTop': '40px'}
                        ),
                        html.P(
                            'Scope, time period, and MEB type follow the filters above.',
                            style={'color': COLORS['text'], 'fontSize': '12px', 'marginBottom': '5px'}
                        ),
                        dcc.Graph(
                            id='mantika-heatmap',
                            style={'height': '550px'}
                        ),
                    ], style={'padding': '20px'})
                ]),
                
                # Rankings Tab (Mantika)
                #dcc.Tab(label='Regional Comparisions', value='rankings', children=[
                #    html.Div([
                #        html.H2('MEB Prices by Mantika', style={'color': COLORS['text']}),
                #        html.Div(id='rankings-table'),
                #    ], style={'padding': '20px'})
                #]),
                
                # Rankings Tab (Mantika)
                dcc.Tab(label='Regional Comparisions', value='rankings', children=[
                    html.Div([
                        html.H2('MEB Prices by Mantika', style={'color': COLORS['text']}),

                        # 🔽 New: MoM vs YoY filter
                        html.Div([
                            html.Label(
                                'Change type:',
                                style={'fontWeight': 'bold', 'marginRight': '10px', 'color': COLORS['text']}
                            ),
                            dcc.RadioItems(
                                id='rankings-change-type',
                                options=[
                                    {'label': ' Month-on-Month', 'value': 'MoM'},
                                    {'label': ' Year-on-Year', 'value': 'YoY'},
                                ],
                                value='MoM',      # default mode
                                inline=True,
                            ),
                        ], style={'marginBottom': '15px'}),

                        html.Div(id='rankings-table'),
                    ], style={'padding': '20px'})
                ]),

                # Commodities Tab (table like screenshot)
                dcc.Tab(label='Commodity Prices', value='commodities', children=[
                    html.Div([
                        html.H2('Commodity Price Analysis', style={'color': COLORS['text']}),
                        html.Div([
                            html.Label(
                                'Category:', 
                                style={'fontWeight': 'bold', 'marginRight': '10px'}
                            ),
                            dcc.RadioItems(
                                id='commodity-category',
                                options=[
                                    {'label': ' Food', 'value': 'Food'},
                                    {'label': ' Non-Food', 'value': 'Non-Food'},
                                    {'label': ' Fuel', 'value': 'Fuel'},
                                ],
                                value='Food',
                                inline=True,
                                style={'marginBottom': '20px'}
                            ),
                        ]),
                        html.Div(id='commodity-table'),
                    ], style={'padding': '20px'})
                ]),
                
                # Export Tab
                dcc.Tab(label='Data Export', value='export', children=[
                    html.Div([
                        html.H2('Data Export', style={'color': COLORS['text']}),
                        html.P('Export current data to Excel format'),
                        html.Button(
                            'Export to Excel',
                            id='export-button', 
                            style={
                                'backgroundColor': COLORS['primary'],
                                'color': 'white',
                                'padding': '10px 20px',
                                'border': 'none',
                                'borderRadius': '4px',
                                'fontSize': '16px',
                                'cursor': 'pointer',
                                'marginTop': '20px'
                            }),
                        html.Div(id='export-status', style={'marginTop': '20px'}),
                    ], style={'padding': '20px'})
                ]),
            ],
            style={'fontFamily': 'Arial, sans-serif'},
        ),
    ], style={'padding': '0 40px 40px 40px'}),
    
    # Footer
    html.Div([
        html.P(
            '© 2025 World Food Programme - Libya Country Office | Research, Assessment and Monitoring (RAM) Unit',
            style={'margin': '0', 'color': COLORS['text'], 'fontSize': '14px', 'textAlign': 'center'}
        ),
    ], style={
        'backgroundColor': COLORS['secondary'],
        'padding': '20px',
        'marginTop': '40px',
        'borderTop': f'1px solid {COLORS["border"]}'
    }),
    
], style={
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh'
})

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output('date-selector', 'options'),
    Output('date-selector', 'value'),
    Input('date-selector', 'id')
)
def populate_date_selector(_):
    """Populate date selector with available dates"""
    engine = get_engine()
    
    query = text("""
        SELECT DISTINCT date 
        FROM national_meb 
        WHERE full_meb IS NOT NULL AND full_meb > 0
        ORDER BY date DESC
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        dates = [row[0] for row in result]
    
    options = [{'label': d.strftime('%B %Y'), 'value': d.isoformat()} for d in dates]
    default_value = dates[0].isoformat() if dates else None
    
    return options, default_value

@app.callback(
    Output('overview-cards', 'children'),
    Input('date-selector', 'value'),
    Input('overview-region-filter', 'value')
)
def update_overview_cards(selected_date, region_scope):
    """Update overview metric cards"""
    if not selected_date:
        return html.Div()
    
    target_date = datetime.fromisoformat(selected_date).date()
    overview = get_national_overview(target_date, region_scope)
    
    if not overview:
        return html.Div('No data available for selected date')
    
    def create_metric_card(title, value, mom_change, color):
        """Create a metric card with MoM change"""
        arrow = '▲' if mom_change > 0 else '▼' if mom_change < 0 else '●'
        change_color = (
            COLORS['warning'] if mom_change > 0
            else COLORS['success'] if mom_change < 0
            else COLORS['text']
        )
        
        return html.Div([
            html.Div(
                title, 
                style={'fontSize': '14px', 'color': COLORS['text'], 'marginBottom': '5px'}
            ),
            html.Div(
                f'LYD {value:,.2f}', 
                style={'fontSize': '32px', 'fontWeight': 'bold', 'color': color}
            ),
            html.Div([
                html.Span(arrow, style={'marginRight': '5px'}),
                html.Span(f'{abs(mom_change):.1f}% MoM', style={'fontSize': '14px'}),
            ], style={'color': change_color, 'marginTop': '5px'}),
        ], style={
            'backgroundColor': COLORS['background'],
            'padding': '20px',
            'borderRadius': '8px',
            'border': f'2px solid {COLORS["border"]}',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'flex': '1',
            'minWidth': '200px'
        })
    
    cards = html.Div([
        create_metric_card('Full MEB', overview['full_meb'], overview['full_meb_mom'], COLORS['primary']),
        create_metric_card('Food MEB', overview['food_meb'], overview['food_meb_mom'], COLORS['primary']),
        create_metric_card('NFI MEB', overview['nfi_meb'], overview['nfi_meb_mom'], COLORS['primary']),
    ], style={
        'display': 'flex',
        'gap': '20px',
        'flexWrap': 'wrap'
    })
    
    return cards

@app.callback(
    Output('municipality-map', 'figure'),
    Input('date-selector', 'value'),
    Input('map-meb-type', 'value'),
    Input('overview-region-filter', 'value')
)
def update_map(selected_date, meb_type, region_scope):
    """Update municipality map"""
    if not selected_date:
        return go.Figure()
    
    target_date = datetime.fromisoformat(selected_date).date()
    df = get_municipality_data(target_date)
    
    if df.empty:
        return go.Figure()
    
    # Filter by scope (National vs region)
    if region_scope != 'National':
        df = df[df['region'] == region_scope]
        if df.empty:
            return go.Figure()
    
    # Map labels
    meb_labels = {
        'full_meb': 'Full MEB',
        'food_meb': 'Food MEB',
        'nfi_meb': 'NFI MEB'
    }
    
    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        size=meb_type,
        color=meb_type,
        hover_name=None,   # we will override hover completely
        custom_data=[
            df['municipality'],
            df['region'],
            df['full_meb'],
            df['food_meb'],
            df['nfi_meb']
        ],
        color_continuous_scale='RdYlGn_r',
        size_max=30,
        zoom=4.5,
        center={'lat': 27, 'lon': 17},
        mapbox_style='carto-positron',
        title=(
            f'{meb_labels[meb_type]} by Municipality '
            f'({region_scope}) - {target_date.strftime("%B %Y")}'
        )
    )
    
    fig.update_layout(
        font=dict(family='Arial', color=COLORS['text']),
        title_font_size=20,
        coloraxis_colorbar=dict(
            title='MEB Price (LYD)',
            tickformat=',.0f'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )
    
    fig.update_traces(
        hovertemplate=
            "<span style='padding:6px; display:block;'>"
            "<b>%{customdata[0]}</b><br>"
            "Region: %{customdata[1]}<br>"
            "<br><b>Full MEB:</b> %{customdata[2]:,.0f} LYD<br>"
            "<b>Food MEB:</b> %{customdata[3]:,.0f} LYD<br>"
            "<b>NFI MEB:</b> %{customdata[4]:,.0f} LYD<br>"
            "</span>"
            "<extra></extra>"
    )

    fig.update_traces(
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.85)",  # semi-transparent
            font_size=14,
            font_color=COLORS['text'],
            bordercolor="white",
            namelength=-1,
            align="left"
        )
    )

    return fig

@app.callback(
    Output('trend-chart', 'figure'),
    Output('trend-footnote', 'children'),
    Input('date-selector', 'value'),
    Input('trend-meb-type', 'value'),
    Input('trend-months', 'value'),
    Input('overview-region-filter', 'value')
)
def update_trend_chart(selected_date, meb_type, months, region_scope):
    """Update MEB trend chart with gap-linked segments and faded connectors."""
    if not selected_date:
        return go.Figure(), ""
    
    nat_df, reg_df = get_trend_data(meb_type, months)
    if nat_df.empty:
        return go.Figure(), ""
    
    # Ensure datetime
    nat_df['date'] = pd.to_datetime(nat_df['date'])
    reg_df['date'] = pd.to_datetime(reg_df['date'])

    # Build continuous monthly index between first and last national data point
    min_date = nat_df['date'].min().replace(day=1)
    max_date = nat_df['date'].max().replace(day=1)
    monthly_index = pd.date_range(start=min_date, end=max_date, freq='MS')

    # National series reindexed on full monthly index
    nat_series = (
        nat_df.set_index('date')[meb_type]
        .reindex(monthly_index)
    )

    nat_segments, nat_bridges, gap_ranges = build_gap_segments(nat_series)

    # Labels
    meb_labels = {
        'full_meb': 'Full MEB',
        'food_meb': 'Food MEB',
        'nfi_meb': 'NFI MEB'
    }

    fig = go.Figure()

    # Regional lines (gap-linked)
    region_colors = [('East', COLORS['east']),
                     ('West', COLORS['west']),
                     ('South', COLORS['south'])]

    for region, color in region_colors:
        # If scope is specific, only draw that region
        if region_scope != 'National' and region != region_scope:
            continue

        region_data = reg_df[reg_df['region'] == region].copy()
        if region_data.empty:
            continue

        reg_series = (
            region_data.set_index('date')[meb_type]
            .reindex(monthly_index)
        )

        reg_segments, reg_bridges, _ = build_gap_segments(reg_series)

        # Real segments
        first_seg = True
        for xs, ys in reg_segments:
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                name=region,
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                showlegend=first_seg  # avoid multiple legend entries per region
            ))
            first_seg = False

        # Faded connectors across gaps
        for x1, y1, x2, y2 in reg_bridges:
            fig.add_trace(go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                opacity=0.2,
                showlegend=False,
                hoverinfo='skip'
            ))

    # National line (gap-linked, dashed)
    first_nat = True
    for xs, ys in nat_segments:
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            name='National',
            mode='lines+markers',
            line=dict(color=COLORS['national'], width=3, dash='dash'),
            marker=dict(size=8),
            showlegend=first_nat
        ))
        first_nat = False

    for x1, y1, x2, y2 in nat_bridges:
        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode='lines',
            line=dict(color=COLORS['national'], width=2, dash='dot'),
            opacity=0.2,
            showlegend=False,
            hoverinfo='skip'
        ))

    # Default: no footnote
    footnote = ""

    # Missing data → footnote text
    if gap_ranges:
        pieces = []
        for start, end in gap_ranges:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            if start.year == end.year and start.month == end.month:
                pieces.append(start.strftime('%b %Y'))
            else:
                pieces.append(f"{start.strftime('%b %Y')} – {end.strftime('%b %Y')}")
        footnote = "No data available for: " + "; ".join(pieces)

    fig.update_layout(
        title='',
        xaxis_title='Month',
        yaxis_title='LYD',
        font=dict(family='Arial', color=COLORS['text']),
        title_font_size=20,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        plot_bgcolor='white',
        margin=dict(l=60, r=20, t=60, b=60)
    )

    fig.update_xaxes(showgrid=True, gridcolor=COLORS['border'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'])
    
    return fig, footnote

@app.callback(
    Output('mantika-heatmap', 'figure'),
    Input('overview-region-filter', 'value'),
    Input('trend-meb-type', 'value'),
    Input('trend-months', 'value'),
)
def update_municipality_heatmap(region_scope, meb_type, months):
    """
    Municipality-level heatmap:
    - Uses same scope & MEB type as trends
    - Time window follows trend-months (6 / 12 / all)
    - Months are ordered chronologically
    - Missing months show as empty gaps (no value)
    """
    df = get_municipality_heatmap_data(
        meb_type=meb_type, scope_region=region_scope
    )

    if df.empty:
        return go.Figure().update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[dict(
                text="No data available",
                x=0.5, y=0.5, showarrow=False
            )],
            margin=dict(l=0, r=0, t=0, b=0)
        )

    # -------------------------------
    # Build full monthly index (for gaps)
    # -------------------------------
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min().replace(day=1)
    max_date = df['date'].max().replace(day=1)
    all_months = pd.date_range(min_date, max_date, freq='MS')

    # Apply time window (6 / 12 / all)
    if months != 999:
        all_months = all_months[-months:]

    # Labels for all months (x-axis)
    month_labels_all = [d.strftime("%Y-%b") for d in all_months]

    # Add month_label to data and keep only months in the selected window
    df['month_label'] = df['date'].dt.strftime("%Y-%b")
    df = df[df['month_label'].isin(month_labels_all)]

    if df.empty:
        return go.Figure().update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[dict(
                text="No data available in selected period",
                x=0.5, y=0.5, showarrow=False
            )],
            margin=dict(l=0, r=0, t=0, b=0)
        )

    # Stable order for municipalities (rows)
    muni_order = sorted(df["municipality"].unique())

    # Pivot to matrix form: rows = municipalities, columns = months
    heat_df = (
        df.pivot(index="municipality",
                 columns="month_label",
                 values="meb_value")
        .reindex(index=muni_order, columns=month_labels_all)
    )

    fig = go.Figure(data=go.Heatmap(
        x=heat_df.columns,
        y=heat_df.index,
        z=heat_df.values,
        colorscale="RdYlGn_r",  # red = high, green = low
        colorbar=dict(title="MEB (LYD)"),
        hovertemplate=(
            "Month: %{x}<br>"
            "Municipality: %{y}<br>"
            "MEB: %{z:.1f}<extra></extra>"
        ),
        xgap=1,
        ygap=1
    ))

    fig.update_layout(
        margin=dict(l=80, r=20, t=20, b=80),
        xaxis=dict(
            title="Month",
            type="category",
            tickangle=-45
        ),
        yaxis=dict(
            title="Municipality",
            autorange="reversed"
        ),
        font=dict(family="Arial", color=COLORS["text"]),
        plot_bgcolor="white"
    )

    return fig


@app.callback(
    Output('rankings-table', 'children'),
    Input('date-selector', 'value'),
    Input('overview-region-filter', 'value'),
    Input('rankings-change-type', 'value')   # 🔽 new input
)
def update_rankings_table(selected_date, region_scope, change_type):
    """
    Update municipality rankings table.

    For each MEB type (Food, NFI, Full), show in the same cell:
        line 1: price (LYD)
        line 2: coloured % change (green/red)

    - Table is filtered by main scope (National/East/West/South).
    - change_type: 'MoM' (previous month) or 'YoY' (same month last year).
    """
    if not selected_date:
        return html.Div()
    
    target_date = datetime.fromisoformat(selected_date).date()
    target_dt = pd.to_datetime(target_date)

    df = get_municipality_data(target_date)
    if df.empty:
        return html.Div('No data available')
    
    # Filter by main scope (National / region)
    if region_scope != 'National':
        df = df[df['region'] == region_scope]
        if df.empty:
            return html.Div('No data available')

    # 🔽 Decide comparison period based on MoM vs YoY
    if change_type == 'YoY':
        # Same month last year
        prev_dt = target_dt.replace(year=target_dt.year - 1)
    else:
        # Month-on-month: previous calendar month
        if target_dt.month == 1:
            prev_dt = target_dt.replace(year=target_dt.year - 1, month=12, day=1)
        else:
            prev_dt = target_dt.replace(month=target_dt.month - 1, day=1)

    # Load previous period municipality data for % change calculation
    engine = get_engine()
    prev_query = text("""
        SELECT 
            m.municipality,
            m.food_meb,
            m.nfi_meb,
            m.full_meb,
            l.adm1_en as region
        FROM municipality_meb m
        JOIN locations l ON m.adm2_pcode = l.adm2_pcode
        WHERE m.date = :date
          AND m.full_meb IS NOT NULL
          AND m.full_meb > 0
    """)
    with engine.connect() as conn:
        prev_result = conn.execute(prev_query, {'date': prev_dt.date()})
        prev_df = pd.DataFrame(
            prev_result.fetchall(),
            columns=['municipality', 'food_meb_prev', 'nfi_meb_prev', 'full_meb_prev', 'region']
        )

    if not prev_df.empty:
        for col in ['food_meb_prev', 'nfi_meb_prev', 'full_meb_prev']:
            prev_df[col] = prev_df[col].astype(float)
        # Merge previous period on municipality + region
        df = df.merge(
            prev_df,
            on=['municipality', 'region'],
            how='left'
        )
    else:
        df['food_meb_prev'] = np.nan
        df['nfi_meb_prev'] = np.nan
        df['full_meb_prev'] = np.nan

    # Calculate % change for each MEB type (numeric helpers)
    df['food_meb_mom'] = np.where(
        (df['food_meb_prev'].notna()) & (df['food_meb_prev'] != 0),
        (df['food_meb'] - df['food_meb_prev']) / df['food_meb_prev'] * 100,
        np.nan
    )
    df['nfi_meb_mom'] = np.where(
        (df['nfi_meb_prev'].notna()) & (df['nfi_meb_prev'] != 0),
        (df['nfi_meb'] - df['nfi_meb_prev']) / df['nfi_meb_prev'] * 100,
        np.nan
    )
    df['full_meb_mom'] = np.where(
        (df['full_meb_prev'].notna()) & (df['full_meb_prev'] != 0),
        (df['full_meb'] - df['full_meb_prev']) / df['full_meb_prev'] * 100,
        np.nan
    )

    if df.empty:
        return html.Div('No data available')

    # Reset index for ranking
    df = df.reset_index(drop=True)
    df.insert(0, 'rank', range(1, len(df) + 1))

    # Build display dataframe
    display_df = df[[
        'rank', 'municipality', 'region',
        'food_meb', 'nfi_meb', 'full_meb',
        'food_meb_mom', 'nfi_meb_mom', 'full_meb_mom'
    ]].copy()

    # Helper to format "price<br>colored % line"
    def format_meb_html(price, pct_change):
        if pd.isna(price):
            return 'N/A'
        base = f'LYD {price:,.2f}'
        if pd.isna(pct_change):
            return base

        arrow = '▲' if pct_change > 0 else '▼' if pct_change < 0 else '●'
        sign = '+' if pct_change > 0 else '' if pct_change < 0 else ''
        color = (
            COLORS['warning'] if pct_change > 0
            else COLORS['success'] if pct_change < 0
            else COLORS['text']
        )
        return (
            f'{base}<br>'
            f'<span style="color:{color}; font-weight:bold;">'
            f'{arrow} {sign}{pct_change:.1f}%'
            f'</span>'
        )

    display_df['food_meb_display'] = display_df.apply(
        lambda r: format_meb_html(r['food_meb'], r['food_meb_mom']), axis=1
    )
    display_df['nfi_meb_display'] = display_df.apply(
        lambda r: format_meb_html(r['nfi_meb'], r['nfi_meb_mom']), axis=1
    )
    display_df['full_meb_display'] = display_df.apply(
        lambda r: format_meb_html(r['full_meb'], r['full_meb_mom']), axis=1
    )

    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[
            {'name': 'Rank', 'id': 'rank'},
            {'name': 'Municipality', 'id': 'municipality'},
            {'name': 'Region', 'id': 'region'},
            {
                'name': 'Food MEB (LYD)',
                'id': 'food_meb_display',
                'presentation': 'markdown'
            },
            {
                'name': 'NFI MEB (LYD)',
                'id': 'nfi_meb_display',
                'presentation': 'markdown'
            },
            {
                'name': 'Full MEB (LYD)',
                'id': 'full_meb_display',
                'presentation': 'markdown'
            },
        ],
        markdown_options={"html": True},
        style_header={
            'backgroundColor': COLORS['primary'],
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'left',
            'padding': '12px'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'fontFamily': 'Arial',
            'fontSize': '14px',
            'color': COLORS['text'],
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': COLORS['secondary']
            },
            {
                'if': {'column_id': 'rank'},
                'fontWeight': 'bold',
                'width': '80px'
            }
        ],
        style_table={'overflowX': 'auto'},
        page_size=25
    )
    
    return table


@app.callback(
    Output('commodity-table', 'children'),
    Input('date-selector', 'value'),
    Input('commodity-category', 'value'),
    Input('overview-region-filter', 'value')
)
def update_commodity_table(selected_date, category, region_scope):
    """
    Commodity prices table similar to Excel screenshot.

    - Two month columns: Previous Month, Selected Month
    - Each cell: price line + MoM% line
        * Selected month MoM = Selected vs Previous
        * Previous month MoM = Previous vs Month-before-previous
    - Filtered by category and main scope (National / East / West / South).
    """
    if not selected_date:
        return html.Div()

    # ---- figure out the three months ----
    target_date = datetime.fromisoformat(selected_date).date()
    target_dt = pd.to_datetime(target_date)

    # previous month (M-1)
    if target_dt.month == 1:
        prev1_dt = target_dt.replace(year=target_dt.year - 1, month=12, day=1)
    else:
        prev1_dt = target_dt.replace(month=target_dt.month - 1, day=1)

    # month before previous (M-2)
    if prev1_dt.month == 1:
        prev2_dt = prev1_dt.replace(year=prev1_dt.year - 1, month=12, day=1)
    else:
        prev2_dt = prev1_dt.replace(month=prev1_dt.month - 1, day=1)

    # ---- load data for the three months ----
    curr_df = get_commodity_data(target_date)
    prev1_df = get_commodity_data(prev1_dt.date())
    prev2_df = get_commodity_data(prev2_dt.date())

    # filter by category
    curr_df = curr_df[curr_df['category'] == category]
    prev1_df = prev1_df[prev1_df['category'] == category]
    prev2_df = prev2_df[prev2_df['category'] == category]

    # filter by scope
    if region_scope == 'National':
        curr_df = curr_df[curr_df['region'] == 'Libya']
        prev1_df = prev1_df[prev1_df['region'] == 'Libya']
        prev2_df = prev2_df[prev2_df['region'] == 'Libya']
    else:
        curr_df = curr_df[curr_df['region'] == region_scope]
        prev1_df = prev1_df[prev1_df['region'] == region_scope]
        prev2_df = prev2_df[prev2_df['region'] == region_scope]

    # if even the selected month has no data, nothing to show
    if curr_df.empty:
        return html.Div('No data available')

    # keep just product + price from each month
    curr_df = curr_df[['product_name', 'average_price']].rename(
        columns={'average_price': 'price_curr'}
    )
    prev1_df = prev1_df[['product_name', 'average_price']].rename(
        columns={'average_price': 'price_prev1'}
    )
    prev2_df = prev2_df[['product_name', 'average_price']].rename(
        columns={'average_price': 'price_prev2'}
    )

    # ---- merge, using CURRENT month as the base ----
    df = curr_df.merge(prev1_df, on='product_name', how='left')
    df = df.merge(prev2_df, on='product_name', how='left')

    # ---- MoM calculations ----
    # M-1 vs M-2 (for previous column)
    df['mom_prev1'] = np.where(
        (df['price_prev1'].notna()) &
        (df['price_prev2'].notna()) &
        (df['price_prev2'] != 0),
        (df['price_prev1'] - df['price_prev2']) / df['price_prev2'] * 100,
        np.nan
    )
    # M vs M-1 (for current column)
    df['mom_curr'] = np.where(
        (df['price_curr'].notna()) &
        (df['price_prev1'].notna()) &
        (df['price_prev1'] != 0),
        (df['price_curr'] - df['price_prev1']) / df['price_prev1'] * 100,
        np.nan
    )

    # sort nicely
    df = df.sort_values('product_name').reset_index(drop=True)

    # column labels
    prev_label = prev1_dt.strftime('%B %Y')
    curr_label = target_dt.strftime('%B %Y')

    # helpers
    def format_price_with_mom(price, mom):
        if pd.isna(price):
            return 'N/A'
        base = f'LYD {price:,.2f}'
        if pd.isna(mom):
            # show price only if we cannot compute MoM
            return base
        arrow = '▲' if mom > 0 else '▼' if mom < 0 else '●'
        sign = '+' if mom > 0 else '' if mom < 0 else ''
        color = (
            COLORS['warning'] if mom > 0
            else COLORS['success'] if mom < 0
            else COLORS['text']
        )
        return (
            f'{base}<br>'
            f'<span style="color:{color}; font-weight:bold;">'
            f'{arrow} {sign}{mom:.1f}%'
            f'</span>'
        )

    display_df = pd.DataFrame({
        'product_name': df['product_name'],
        # previous month: price_prev1 with MoM vs prev2 (if prev2 exists)
        'prev_display': [
            format_price_with_mom(p, m)
            for p, m in zip(df['price_prev1'], df['mom_prev1'])
        ],
        # selected month: price_curr with MoM vs prev1 (if prev1 exists)
        'curr_display': [
            format_price_with_mom(p, m)
            for p, m in zip(df['price_curr'], df['mom_curr'])
        ]
    })

    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[
            {'name': 'Product', 'id': 'product_name'},
            {'name': prev_label, 'id': 'prev_display', 'presentation': 'markdown'},
            {'name': curr_label, 'id': 'curr_display', 'presentation': 'markdown'},
        ],
        markdown_options={"html": True},
        style_header={
            'backgroundColor': COLORS['primary'],
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'left',
            'padding': '12px'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '8px 12px',
            'fontFamily': 'Arial',
            'fontSize': '14px',
            'color': COLORS['text'],
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': COLORS['secondary']
            }
        ],
        style_table={'overflowX': 'auto'},
        page_size=50
    )

    return table


@app.callback(
    Output('export-status', 'children'),
    Input('export-button', 'n_clicks'),
    State('date-selector', 'value'),
    prevent_initial_call=True
)
def export_data(n_clicks, selected_date):
    """Export current data to Excel"""
    if not selected_date:
        return html.Div('Please select a date', style={'color': COLORS['warning']})
    
    try:
        target_date = datetime.fromisoformat(selected_date).date()
        
        # Get data
        muni_df = get_municipality_data(target_date)
        commodity_df = get_commodity_data(target_date)
        
        # Create export
        output_dir = Path('/mnt/user-data/outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f'PMM_Export_{target_date.strftime("%b%y")}.xlsx'
        output_file = output_dir / filename
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            muni_df.to_excel(writer, sheet_name='Municipalities', index=False)
            commodity_df.to_excel(writer, sheet_name='Commodities', index=False)
        
        return html.Div([
            html.P('✅ Export successful!', style={'color': COLORS['success'], 'fontWeight': 'bold'}),
            html.A(
                f'Download {filename}', 
                href=f'computer:///mnt/user-data/outputs/{filename}',
                style={'color': COLORS['primary'], 'textDecoration': 'underline'}
            )
        ])
        
    except Exception as e:
        return html.Div(f'❌ Export failed: {str(e)}', style={'color': COLORS['warning']})

# ============================================================================
# RUN SERVER
# ============================================================================
server = app.server

@app.server.route('/healthz')
def healthcheck():
    return "ok", 200

if __name__ == '__main__':
    print("="*70)
    print("LIBYA PMM DASHBOARD")
    print("="*70)
    print("\n🚀 Starting dashboard server...")
    print("\n📊 Open your browser to: http://127.0.0.1:8051")
    print("\n⏹  Press Ctrl+C to stop the server")
    print("="*70)
    
    app.run_server(debug=True, host='127.0.0.1', port=8051)
