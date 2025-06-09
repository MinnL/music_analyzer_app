import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import base64
from io import BytesIO
import os
import json
import time

from app.audio_input import AudioInput
from app.analysis import MusicAnalyzer
from app.visualization import create_rhythm_visualization, create_melody_visualization, create_instrumentation_visualization

# Initialize Dash app
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
                ],
                assets_folder='app/static')
server = app.server
app.title = "Genra - Music Analysis & Genre Classification"  # Update the browser tab title too

# Initialize audio input and analyzer with matching sample rates
audio_input = AudioInput(sample_rate=22050, demo_mode_if_error=True)  # Match analyzer sample rate
# Configure to use the original GenreClassifier with GTZAN weights
music_analyzer = MusicAnalyzer(use_gtzan_model=True, sample_rate=audio_input.sample_rate) 

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Genra", className="text-center display-3 mt-4 mb-0"),
            html.H4("Discover the Science Behind Your Sound", className="text-center text-muted mb-4"),
            html.Div([
                html.P("Genra analyzes your music in real-time, identifying genres and breaking down the rhythm, melody, and instrumentation components that make your sound unique.", 
                       className="text-center lead mb-4")
            ], className="px-4 mb-3")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Audio Input Controls", className="section-title"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Start Recording", id="record-button", color="success", className="me-2"),
                            dbc.Button("Stop Recording", id="stop-button", color="danger", className="ms-2", disabled=True),
                        ], width=12, className="mb-3")
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Or upload an audio file:", className="mb-2"),
                            dcc.Upload(
                                id='upload-audio',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select an Audio File')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px 0'
                                },
                                multiple=False
                            ),
                            html.Div(id="upload-status", className="mt-2"),
                        ], width=12)
                    ]),
                    html.Div(id="recording-status", className="mt-2"),
                    dbc.Alert(
                        "Demo Mode: Using synthesized audio instead of microphone",
                        id="demo-mode-alert",
                        color="warning",
                        is_open=False,
                        className="mt-3"
                    ),
                    dbc.Alert(
                        id="file-mode-alert",
                        color="info",
                        is_open=False,
                        className="mt-3"
                    ),
                    # Add recording timer display
                    dbc.Alert(
                        [
                            html.Span("Recording time: "),
                            html.Span("0:00", id="recording-timer", className="fw-bold")
                        ],
                        id="timer-display",
                        color="danger",
                        is_open=False,
                        className="mt-3 d-flex align-items-center"
                    ),
                    # Add loading indicator
                    dbc.Alert(
                        "Analyzing audio... This may take a moment for longer recordings.",
                        id="analyzing-alert",
                        color="secondary",
                        is_open=False,
                        className="mt-3"
                    ),
                    # Add audio playback section
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dbc.Button(
                                    [html.I(className="fas fa-play me-1"), "Play Audio"], 
                                    id="play-button", 
                                    color="primary", 
                                    className="mt-3",
                                    disabled=True
                                ),
                                html.Div(id="playback-info", className="mt-2 text-muted fst-italic")
                            ], id="playback-controls", className="d-flex flex-column align-items-center"),
                            # Hidden audio component
                            html.Div(id="audio-player-container", className="mt-3 text-center")
                        ], width=12)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Genre Classification", className="section-title"),
                dbc.CardBody([
                    html.H3(id="genre-display", className="text-center"),
                    html.Div(id="genre-confidence", className="text-center"),
                    # Add the genre explanation section
                    html.Div(id="genre-explanation", className="mt-4")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Add instrument detection section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Instrument Detection", className="section-title"),
                dbc.CardBody([
                    html.Div(id="instrument-details")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rhythm Analysis", className="section-title"),
                dbc.CardBody([
                    dcc.Graph(id="rhythm-graph")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Melody Analysis", className="section-title"),
                dbc.CardBody([
                    dcc.Graph(id="melody-graph")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Instrumentation Analysis", className="section-title"),
                dbc.CardBody([
                    dcc.Graph(id="instrumentation-graph")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Component Details", className="section-title"),
                dbc.CardBody([
                    html.Div(id="component-details")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Add available genres section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Available Genres", className="section-title"),
                dbc.CardBody([
                    # Collapsible section for available genres
                    dbc.Collapse(
                        id="genres-collapse",
                        is_open=False,
                        children=[
                            html.Div(id="available-genres")
                        ]
                    ),
                    # Button to toggle collapse
                    dbc.Button(
                        "Show Available Genres",
                        id="genres-button",
                        className="mt-2",
                        color="info"
                    )
                ])
            ])
        ], width=12)
    ]),
    
    # Debug mode indicator
    html.Div(
        html.Span(id="debug-indicator", className="debug-dot"),
        style={"position": "fixed", "bottom": "5px", "right": "5px", "zIndex": 1000}
    ),
    
    # Hidden elements for storing state
    dcc.Store(id="audio-data"),
    dcc.Store(id="demo-mode-store", data=False),
    dcc.Store(id="file-mode-store", data=False),
    dcc.Store(id="audio-playback-data", data=None),
    dcc.Store(id="last-analysis-time", data=0),
    dcc.Store(id="analysis-needed", data=False),
    dcc.Store(id="analysis-results", data=None),  # Store analysis results for visualization updates
    dcc.Store(id="recording-time", data={"start_time": 0, "recording": False}),  # Store for recording timer
    # Disabled by default - only used for file mode
    dcc.Interval(id="update-interval", interval=2000, n_intervals=0, disabled=True),  # Only used for file mode
    # Timer update interval - updates the recording timer display
    dcc.Interval(id="timer-interval", interval=100, n_intervals=0, disabled=True),  # 100ms for smooth timer updates
    # Add a debug mode store component
    dcc.Store(id="debug-instrument-mode", data=False),
], fluid=True)

# Callbacks
@callback(
    [Output("record-button", "disabled"),
     Output("stop-button", "disabled"),
     Output("update-interval", "disabled"),
     Output("recording-status", "children"),
     Output("demo-mode-alert", "is_open"),
     Output("demo-mode-store", "data"),
     Output("file-mode-alert", "is_open"),
     Output("file-mode-store", "data"),
     Output("file-mode-alert", "children"),
     Output("analysis-needed", "data"),
     Output("timer-interval", "disabled"),
     Output("timer-display", "is_open"),
     Output("recording-time", "data")],
    [Input("record-button", "n_clicks"),
     Input("stop-button", "n_clicks")],
    [State("demo-mode-store", "data"),
     State("file-mode-store", "data")],
    prevent_initial_call=True
)
def toggle_recording(start_clicks, stop_clicks, demo_mode, file_mode):
    triggered_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_id == "record-button":
        audio_input.start_recording()
        print("Recording started... (up to 60 seconds)")
        # Check if we're in demo mode
        demo_mode = audio_input.demo_mode
        status_text = "Recording in progress..." if not demo_mode else "Demo mode active (using synthesized audio)"
        # Start the timer
        timer_data = {"start_time": time.time(), "recording": True}
        # When starting recording, exit file mode and don't analyze yet
        return True, False, True, status_text, demo_mode, demo_mode, False, False, "", False, False, True, timer_data
    elif triggered_id == "stop-button":
        print("Stopping recording and saving audio...")
        audio_input.stop_recording()
        print("Recording stopped...")
        # Make sure audio is saved for playback
        print(f"Audio available for playback: {audio_input.current_audio is not None}")
        # Stop the timer
        timer_data = {"start_time": 0, "recording": False}
        # Signal that we need to analyze the audio
        return False, True, True, "Recording stopped. Analyzing audio...", False, False, False, False, "", True, True, False, timer_data
    
    return False, True, True, "", False, False, False, False, "", False, True, False, {"start_time": 0, "recording": False}

@callback(
    [Output("upload-status", "children"),
     Output("file-mode-alert", "is_open", allow_duplicate=True),
     Output("file-mode-store", "data", allow_duplicate=True),
     Output("file-mode-alert", "children", allow_duplicate=True),
     Output("record-button", "disabled", allow_duplicate=True),
     Output("stop-button", "disabled", allow_duplicate=True),
     Output("update-interval", "disabled", allow_duplicate=True),
     Output("analyzing-alert", "is_open"),
     Output("analysis-needed", "data", allow_duplicate=True)],
    [Input("upload-audio", "contents")],
    [State("upload-audio", "filename")],
    prevent_initial_call=True
)
def process_uploaded_file(contents, filename):
    if contents is None:
        return "", False, False, "", False, True, True, False, False
    
    # Show analyzing alert
    success, message = audio_input.process_uploaded_file(contents, filename)
    
    if success:
        alert_message = f"Analyzing file: {filename}"
        # For file uploads, keep interval disabled but trigger analysis via analysis-needed flag
        return message, True, True, alert_message, True, True, True, True, True
    else:
        return message, False, False, "", False, True, True, False, False

@callback(
    [Output("genre-display", "children"),
     Output("genre-confidence", "children"),
     Output("rhythm-graph", "figure"),
     Output("melody-graph", "figure"),
     Output("instrumentation-graph", "figure"),
     Output("play-button", "disabled"),
     Output("analyzing-alert", "is_open", allow_duplicate=True),
     Output("last-analysis-time", "data"),
     Output("analysis-needed", "data", allow_duplicate=True),
     Output("recording-status", "children", allow_duplicate=True),
     Output("analysis-results", "data"),
     Output("genre-explanation", "children"),  # Genre explanation
     Output("available-genres", "children"),   # Available genres
     Output("instrument-details", "children")],  # Instrument detection - new addition
    [Input("update-interval", "n_intervals"),
     Input("analysis-needed", "data")],
    [State("file-mode-store", "data"),
     State("last-analysis-time", "data"),
     State("analysis-results", "data"),
     State("genres-collapse", "is_open"),
     State("debug-instrument-mode", "data")],  # Add debug mode state
    prevent_initial_call=True
)
def update_analysis(n_intervals, analysis_needed, file_mode, last_analysis_time, 
                   prev_analysis_results, genres_open, debug_instruments):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # First check if analysis is needed due to recording stop
    if triggered_id == "analysis-needed" and analysis_needed:
        print("Performing analysis after recording stopped")
        # Small delay to ensure audio data is saved after recording stops
        time.sleep(0.2)
        analyzing_alert_visible = True
    # Or if we're in file mode with interval updates
    elif triggered_id == "update-interval":
        if not file_mode:
            return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    dash.no_update, dash.no_update, last_analysis_time, False, dash.no_update, 
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update)
    
    # Only analyze if enough time has passed since last analysis (3 seconds minimum)
        if time.time() - last_analysis_time < 3 and last_analysis_time > 0:
        # Return previous state without updating
            return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    dash.no_update, dash.no_update, last_analysis_time, False, dash.no_update, 
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update)
            
        analyzing_alert_visible = True
    else:
        # No need to analyze
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                dash.no_update, dash.no_update, last_analysis_time, False, dash.no_update, 
                dash.no_update, dash.no_update, dash.no_update, dash.no_update)
    
    # Get latest audio data
    audio_data = audio_input.get_latest_data()
    
    print(f"DEBUG: audio_data type: {type(audio_data)}, length: {len(audio_data) if audio_data is not None else 'None'}")
    
    if audio_data is None or len(audio_data) == 0:
        empty_result = {
            "genre": "No data",
            "confidence": 0,
            "components": {
                "rhythm": {},
                "melody": {},
                "instrumentation": {},
                "instruments": []  # Empty instruments list for initial state
            }
        }
        # Get available genres information regardless of analysis
        available_genres = music_analyzer.get_available_genres()
        
        return ("No data", "", go.Figure(), go.Figure(), go.Figure(), True, False, time.time(), 
                False, "No audio data to analyze", empty_result, "No analysis available", available_genres,
                "No instrument data available")
    
    # Analyze audio
    print(f"Analyzing {len(audio_data) / audio_input.sample_rate:.2f} seconds of audio...")
    genre, confidence, components = music_analyzer.analyze(audio_data, debug_instruments=debug_instruments)
    print(f"Analysis complete. Genre: {genre}, Confidence: {confidence:.2f}%")
    
    # Store the results for visualization
    analysis_results = {
        "genre": genre,
        "confidence": confidence,
        "components": components
    }
    
    # Create visualizations
    rhythm_fig = create_rhythm_visualization(components['rhythm'])
    melody_fig = create_melody_visualization(components['melody'])
    instrumentation_fig = create_instrumentation_visualization(components['instrumentation'])
    
    confidence_text = f"Confidence: {confidence:.2f}%"
    
    # Display source
    if audio_input.file_mode:
        genre_display = f"{genre} (File: {audio_input.file_info['filename']})"
    elif audio_input.demo_mode:
        genre_display = f"{genre} (Demo)"
    else:
        genre_display = genre
    
    # Enable play button if we have audio data and either:
    # 1. We have current audio saved
    # 2. We're in file mode
    play_button_disabled = True
    
    # Check if we have current audio for playback
    if audio_input.current_audio is not None:
        print(f"Current audio available: {len(audio_input.current_audio)} samples, {len(audio_input.current_audio)/audio_input.sample_rate:.2f} seconds")
        play_button_disabled = False
    elif audio_input.file_mode:
        play_button_disabled = False
    else:
        print("No audio available for playback")
        play_button_disabled = True
    
    # Hide analyzing alert
    analyzing_alert_visible = False
    
    # Reset analysis needed flag and update the status text
    status_text = f"Analysis complete. Detected genre: {genre} (Confidence: {confidence:.2f}%). Click Play Audio to listen."
    
    # Generate genre explanation
    genre_explanation = music_analyzer.get_genre_explanation(genre, components)
    
    # Get available genres information
    available_genres = music_analyzer.get_available_genres()
    
    # Get instrument details - new addition
    instruments = components.get('instruments', [])
    instrument_details = music_analyzer.get_instrument_details(instruments, genre)
    
    return (genre_display, confidence_text, rhythm_fig, melody_fig, instrumentation_fig, 
            play_button_disabled, analyzing_alert_visible, time.time(), False, status_text, 
            analysis_results, genre_explanation, available_genres, instrument_details)

@callback(
    [Output("audio-playback-data", "data"),
     Output("playback-info", "children")],
    [Input("play-button", "n_clicks"),
     Input("stop-button", "n_clicks")],
    [State("audio-playback-data", "data")],
    prevent_initial_call=True
)
def prepare_audio_playback(play_clicks, stop_clicks, current_playback_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None, ""
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # If stop button is clicked, we don't need to save the audio here
    # as it's already saved in the toggle_recording callback
    if triggered_id == "stop-button":
        return None, ""
    
    # If play button is clicked, get audio for playback
    if triggered_id == "play-button":
        print("Play button clicked, retrieving audio...")
        playback_data = audio_input.get_audio_for_playback()
        if playback_data:
            info_text = f"Playing {playback_data['source']} ({playback_data['duration']:.2f} seconds)"
            print(f"Playing back audio: {info_text}")
            return playback_data, info_text
        else:
            print("No audio available for playback")
            return None, "No audio available for playback"
            
    return current_playback_data, ""

@callback(
    Output("audio-player-container", "children"),
    [Input("audio-playback-data", "data")],
    prevent_initial_call=True
)
def update_audio_player(playback_data):
    if not playback_data or "data" not in playback_data:
        return []
    
    # Create an HTML audio element
    audio_player = html.Audio(
        id="audio-player",
        src=playback_data["data"],
        controls=True,
        autoPlay=True,
        className="w-100"
    )
    
    return [audio_player]

@callback(
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

@callback(
    Output("play-button", "disabled", allow_duplicate=True),
    [Input("stop-button", "n_clicks")],
    prevent_initial_call=True
)
def enable_play_after_stop(n_clicks):
    print("Stop button clicked, enabling play button")
    # Check if audio is actually available
    if audio_input.current_audio is not None:
        print(f"Audio is available for playback: {len(audio_input.current_audio)} samples")
        return False
    else:
        print("No audio available for playback after stopping")
        return True

@callback(
    [Output("rhythm-graph", "figure", allow_duplicate=True),
     Output("melody-graph", "figure", allow_duplicate=True),
     Output("instrumentation-graph", "figure", allow_duplicate=True)],
    [Input("analysis-results", "data")],
    prevent_initial_call=True
)
def update_visualizations(analysis_results):
    if not analysis_results or "components" not in analysis_results:
        # Return empty figures if no data
        return go.Figure(), go.Figure(), go.Figure()
    
    # Create visualizations from stored analysis results
    components = analysis_results["components"]
    rhythm_fig = create_rhythm_visualization(components.get('rhythm', {}))
    melody_fig = create_melody_visualization(components.get('melody', {}))
    instrumentation_fig = create_instrumentation_visualization(components.get('instrumentation', {}))
    
    print("Visualization graphs updated from analysis results")
    return rhythm_fig, melody_fig, instrumentation_fig

@callback(
    Output("analysis-needed", "data", allow_duplicate=True),
    [Input("stop-button", "n_clicks")],
    prevent_initial_call=True
)
def stop_recording_and_analyze(n_clicks):
    print("Stop button clicked - triggering analysis...")
    return True

@callback(
    Output("recording-timer", "children"),
    [Input("timer-interval", "n_intervals")],
    [State("recording-time", "data")],
    prevent_initial_call=True
)
def update_recording_timer(n_intervals, timer_data):
    if not timer_data or not timer_data.get("recording", False):
        return "0:00"
    
    # Calculate elapsed time
    elapsed_seconds = time.time() - timer_data.get("start_time", 0)
    
    # Format as minutes:seconds
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    
    # Return formatted time
    return f"{minutes}:{seconds:02d}"

# Add new callback for genres section toggle
@callback(
    [Output("genres-collapse", "is_open"),
     Output("genres-button", "children")],
    [Input("genres-button", "n_clicks")],
    [State("genres-collapse", "is_open")],
    prevent_initial_call=True
)
def toggle_genres_section(n_clicks, is_open):
    if n_clicks:
        # Toggle the collapse state
        new_state = not is_open
        # Update the button text based on state
        button_text = "Hide Available Genres" if new_state else "Show Available Genres"
        return new_state, button_text
    return is_open, "Show Available Genres"

# Add a callback to toggle debug mode for instrument detection
@callback(
    Output("debug-instrument-mode", "data"),
    Input("play-button", "n_clicks"),
    State("debug-instrument-mode", "data"),
    prevent_initial_call=True
)
def toggle_debug_mode(n_clicks, current_debug_state):
    """Toggle debug mode for instrument detection with a special button combination"""
    # Check if this is a special click pattern (double click)
    if n_clicks and n_clicks % 5 == 0:  # Every 5th click
        return not current_debug_state
    return current_debug_state

# Add callback for debug indicator
@callback(
    Output("debug-indicator", "style"),
    Input("debug-instrument-mode", "data")
)
def update_debug_indicator(debug_mode):
    if debug_mode:
        return {"width": "8px", "height": "8px", "borderRadius": "50%", "backgroundColor": "red", "display": "inline-block"}
    else:
        return {"width": "8px", "height": "8px", "borderRadius": "50%", "backgroundColor": "transparent", "display": "inline-block"}

if __name__ == "__main__":
    app.run(debug=True, port=8053) 