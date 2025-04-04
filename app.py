import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import base64
from io import BytesIO
import os

from app.audio_input import AudioInput
from app.analysis import MusicAnalyzer
from app.visualization import create_rhythm_visualization, create_melody_visualization, create_instrumentation_visualization

# Initialize Dash app
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                assets_folder='app/static')
server = app.server

# Initialize audio input and analyzer
audio_input = AudioInput()
music_analyzer = MusicAnalyzer()

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Real-Time Music Analysis & Visualization", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Audio Input Controls"),
                dbc.CardBody([
                    dbc.Button("Start Recording", id="record-button", color="success", className="mr-2"),
                    dbc.Button("Stop Recording", id="stop-button", color="danger", className="ml-2", disabled=True),
                    html.Div(id="recording-status", className="mt-2")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Genre Classification"),
                dbc.CardBody([
                    html.H3(id="genre-display", className="text-center"),
                    html.Div(id="genre-confidence", className="text-center")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rhythm Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="rhythm-graph")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Melody Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="melody-graph")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Instrumentation Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="instrumentation-graph")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Component Details"),
                dbc.CardBody([
                    html.Div(id="component-details")
                ])
            ])
        ], width=12)
    ]),
    
    # Hidden elements for storing state
    dcc.Store(id="audio-data"),
    dcc.Interval(id="update-interval", interval=1000, n_intervals=0, disabled=True)
], fluid=True)

# Callbacks
@app.callback(
    [Output("record-button", "disabled"),
     Output("stop-button", "disabled"),
     Output("update-interval", "disabled"),
     Output("recording-status", "children")],
    [Input("record-button", "n_clicks"),
     Input("stop-button", "n_clicks")],
    prevent_initial_call=True
)
def toggle_recording(start_clicks, stop_clicks):
    triggered_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_id == "record-button":
        audio_input.start_recording()
        return True, False, False, "Recording in progress..."
    elif triggered_id == "stop-button":
        audio_input.stop_recording()
        return False, True, True, "Recording stopped."
    
    return False, True, True, ""

@app.callback(
    [Output("genre-display", "children"),
     Output("genre-confidence", "children"),
     Output("rhythm-graph", "figure"),
     Output("melody-graph", "figure"),
     Output("instrumentation-graph", "figure")],
    [Input("update-interval", "n_intervals")],
    prevent_initial_call=True
)
def update_analysis(n_intervals):
    # Get latest audio data
    audio_data = audio_input.get_latest_data()
    
    if audio_data is None or len(audio_data) == 0:
        return "No data", "", go.Figure(), go.Figure(), go.Figure()
    
    # Analyze audio
    genre, confidence, components = music_analyzer.analyze(audio_data)
    
    # Create visualizations
    rhythm_fig = create_rhythm_visualization(components['rhythm'])
    melody_fig = create_melody_visualization(components['melody'])
    instrumentation_fig = create_instrumentation_visualization(components['instrumentation'])
    
    confidence_text = f"Confidence: {confidence:.2f}%"
    
    return genre, confidence_text, rhythm_fig, melody_fig, instrumentation_fig

@app.callback(
    Output("component-details", "children"),
    [Input("rhythm-graph", "clickData"),
     Input("melody-graph", "clickData"),
     Input("instrumentation-graph", "clickData")]
)
def display_component_details(rhythm_click, melody_click, instrumentation_click):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return "Click on a visualization element to see details."
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    click_data = ctx.triggered[0]["value"]
    
    if trigger_id == "rhythm-graph":
        if click_data:
            # Get details about the rhythm component that was clicked
            component_type = "rhythm"
            # Extract index or value from click_data to identify specific component
            return music_analyzer.get_component_details(component_type, click_data)
    
    elif trigger_id == "melody-graph":
        if click_data:
            # Get details about the melody component that was clicked
            component_type = "melody"
            return music_analyzer.get_component_details(component_type, click_data)
    
    elif trigger_id == "instrumentation-graph":
        if click_data:
            # Get details about the instrumentation component that was clicked
            component_type = "instrumentation"
            return music_analyzer.get_component_details(component_type, click_data)
    
    return "Click on a visualization element to see details."

if __name__ == "__main__":
    app.run_server(debug=True) 