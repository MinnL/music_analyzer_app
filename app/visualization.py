import plotly.graph_objects as go
import numpy as np

def create_rhythm_visualization(rhythm_data):
    """
    Create visualization for rhythm components
    
    Args:
        rhythm_data: Dictionary containing rhythm analysis data
        
    Returns:
        Plotly figure object
    """
    # Check if we have valid data
    if not rhythm_data or "tempo" not in rhythm_data or rhythm_data["tempo"] == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No rhythm data available",
            xaxis_title="Time",
            yaxis_title="Intensity"
        )
        return fig
    
    # Get data from rhythm analysis
    tempo = rhythm_data.get("tempo", 0)
    # Ensure tempo is a scalar value, not an array
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo.item() if tempo.size == 1 else tempo.mean())
    
    tempo_category = rhythm_data.get("tempo_category", "Unknown")
    complexity = rhythm_data.get("complexity", 0)
    complexity_category = rhythm_data.get("complexity_category", "Unknown")
    
    # Create visualization based on tempo and complexity
    # Generate a simple rhythm pattern based on tempo
    seconds = 4  # visualize 4 seconds
    sr = 100  # points per second for visualization
    t = np.linspace(0, seconds, seconds * sr)
    
    # Generate basic beat pattern based on tempo
    # 60 BPM = 1 beat per second, so beats occur every (60/tempo) seconds
    beat_interval = 60 / tempo
    
    # Basic pulse with some randomness for complexity
    pulse = np.zeros_like(t)
    for i in range(int(seconds / beat_interval)):
        beat_time = i * beat_interval
        beat_idx = int(beat_time * sr)
        if beat_idx < len(pulse):
            # Main beat
            pulse[beat_idx:beat_idx+10] = 1.0
            
            # Add some complexity based on the complexity value
            if complexity > 0.4:  # Add some off-beats for moderate+ complexity
                half_beat_idx = int((beat_time + beat_interval/2) * sr)
                if half_beat_idx < len(pulse):
                    pulse[half_beat_idx:half_beat_idx+5] = 0.7 * min(1, complexity)
                    
            if complexity > 0.7:  # Add even more subdivisions for high complexity
                quarter_beat_idx1 = int((beat_time + beat_interval/4) * sr)
                quarter_beat_idx3 = int((beat_time + 3*beat_interval/4) * sr)
                if quarter_beat_idx1 < len(pulse):
                    pulse[quarter_beat_idx1:quarter_beat_idx1+3] = 0.5 * min(1, complexity)
                if quarter_beat_idx3 < len(pulse):
                    pulse[quarter_beat_idx3:quarter_beat_idx3+3] = 0.5 * min(1, complexity)
    
    # Create figure
    fig = go.Figure()
    
    # Add pulse visualization
    fig.add_trace(go.Scatter(
        x=t,
        y=pulse,
        mode='lines',
        line=dict(width=2, color='rgba(49, 130, 189, 1)'),
        name='Beat Pattern',
        hoverinfo='skip'
    ))
    
    # Add beat markers
    beat_times = [i * beat_interval for i in range(int(seconds / beat_interval) + 1) if i * beat_interval <= seconds]
    beat_markers = [1.0] * len(beat_times)
    
    # Create a formatted tempo string for display, ensuring it's a regular Python float
    tempo_display = f"{tempo:.1f}"
    
    fig.add_trace(go.Scatter(
        x=beat_times,
        y=beat_markers,
        mode='markers',
        marker=dict(size=15, color='rgba(255, 0, 0, 0.8)'),
        name='Beat Markers',
        customdata=["tempo"] * len(beat_times),  # Store feature name for component details
        hovertemplate='Beat at %{x:.2f}s<br>Tempo: ' + tempo_display + ' BPM<extra></extra>',
        text=[tempo_display] * len(beat_times)
    ))
    
    # Add annotations
    annotations = [
        dict(
            x=0.5,
            y=1.15,
            xref="paper",
            yref="paper",
            text=f"Tempo: {tempo_display} BPM ({tempo_category})",
            showarrow=False,
            font=dict(size=14)
        ),
        dict(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"Complexity: {complexity_category}",
            showarrow=False,
            font=dict(size=14)
        )
    ]
    
    # Update layout
    fig.update_layout(
        title="Rhythm Analysis",
        xaxis_title="Time (seconds)",
        yaxis_title="Beat Intensity",
        hovermode="closest",
        annotations=annotations,
        margin=dict(l=20, r=20, t=80, b=20),
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(245, 245, 245, 1)',
        paper_bgcolor='rgba(245, 245, 245, 1)'
    )
    
    # Remove y-axis ticks and grid for cleaner look
    fig.update_yaxes(showticklabels=False, showgrid=False)
    
    return fig

def create_melody_visualization(melody_data):
    """
    Create visualization for melody components
    
    Args:
        melody_data: Dictionary containing melody analysis data
        
    Returns:
        Plotly figure object
    """
    # Check if we have valid data
    if not melody_data or not melody_data.get("dominant_notes"):
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No melody data available",
            xaxis_title="Note",
            yaxis_title="Prominence"
        )
        return fig
    
    # Full chromatic scale for reference
    all_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Get melody data
    dominant_notes = melody_data.get("dominant_notes", [])
    pitch_variety = melody_data.get("pitch_variety", 0.0)
    variety_category = melody_data.get("variety_category", "Unknown")
    modality = melody_data.get("modality", "Unknown")
    
    # Ensure pitch_variety is a scalar float, not a numpy array
    if isinstance(pitch_variety, np.ndarray):
        pitch_variety = float(pitch_variety.item() if pitch_variety.size == 1 else pitch_variety.mean())
    
    # Create bar chart of note prominence
    # If we have dominant notes, create values for those notes
    if dominant_notes:
        # Create a value for each note in the chromatic scale
        note_values = []
        for note in all_notes:
            if note in dominant_notes:
                # Higher value for dominant notes, with first one highest
                rank = dominant_notes.index(note)
                note_values.append(float(1.0 - (rank * 0.2)))
            else:
                # Small random value for non-dominant notes
                note_values.append(float(np.random.uniform(0.05, 0.15)))
    else:
        # Dummy data if no notes available
        note_values = [0.1] * len(all_notes)
    
    # Create figure
    fig = go.Figure()
    
    # Choose color based on modality
    if modality == "Major":
        bar_color = 'rgba(255, 165, 0, 0.7)'  # Orange for major
    elif modality == "Minor":
        bar_color = 'rgba(106, 90, 205, 0.7)'  # Purple for minor
    else:
        bar_color = 'rgba(100, 100, 100, 0.7)'  # Gray for unknown
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=all_notes,
        y=note_values,
        marker_color=bar_color,
        customdata=["modality"] * len(all_notes),  # Store feature name for component details
        hovertemplate='Note: %{x}<br>Prominence: %{y:.2f}<extra></extra>'
    ))
    
    # Highlight the scale degrees if we know the modality
    if modality != "Unknown" and dominant_notes:
        root_note = dominant_notes[0]
        root_idx = all_notes.index(root_note)
        
        scale_indices = []
        if modality == "Major":
            # Major scale degrees relative to root: root, M2, M3, P4, P5, M6, M7
            intervals = [0, 2, 4, 5, 7, 9, 11]
        else:  # Minor
            # Minor scale degrees relative to root: root, M2, m3, P4, P5, m6, m7
            intervals = [0, 2, 3, 5, 7, 8, 10]
            
        scale_indices = [(root_idx + interval) % 12 for interval in intervals]
        scale_notes = [all_notes[idx] for idx in scale_indices]
        
        # Add markers for scale degrees
        fig.add_trace(go.Scatter(
            x=scale_notes,
            y=[float(note_values[all_notes.index(note)] + 0.05) for note in scale_notes],
            mode='markers',
            marker=dict(
                symbol='star',
                size=12,
                color='rgba(255, 0, 0, 0.8)',
                line=dict(width=1, color='rgba(0, 0, 0, 1)')
            ),
            name='Scale Degrees',
            hoverinfo='skip'
        ))
    
    # Add annotations
    annotations = [
        dict(
            x=0.5,
            y=1.15,
            xref="paper",
            yref="paper",
            text=f"Modality: {modality}",
            showarrow=False,
            font=dict(size=14)
        ),
        dict(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"Pitch Variety: {variety_category}",
            showarrow=False,
            font=dict(size=14)
        )
    ]
    
    # Add dominant note annotation if available
    if dominant_notes:
        annotations.append(
            dict(
                x=0.5,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Key Center: {dominant_notes[0]}",
                showarrow=False,
                font=dict(size=14)
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Melody Analysis",
        xaxis_title="Note",
        yaxis_title="Prominence",
        annotations=annotations,
        margin=dict(l=20, r=20, t=80, b=20),
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(245, 245, 245, 1)',
        paper_bgcolor='rgba(245, 245, 245, 1)'
    )
    
    # Set y-axis range
    fig.update_yaxes(range=[0, 1.2])
    
    return fig

def create_instrumentation_visualization(instrumentation_data):
    """
    Create visualization for instrumentation components
    
    Args:
        instrumentation_data: Dictionary containing instrumentation analysis data
        
    Returns:
        Plotly figure object
    """
    # Check if we have valid data
    if not instrumentation_data or "brightness" not in instrumentation_data:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No instrumentation data available",
            xaxis_title="Feature",
            yaxis_title="Value"
        )
        return fig
    
    # Get instrumentation data
    brightness = instrumentation_data.get("brightness", 0.0)
    brightness_category = instrumentation_data.get("brightness_category", "Unknown")
    contrast = instrumentation_data.get("contrast", 0.0)
    contrast_category = instrumentation_data.get("contrast_category", "Unknown")
    timbre_complexity = instrumentation_data.get("timbre_complexity", 0.0)
    complexity_category = instrumentation_data.get("complexity_category", "Unknown")
    
    # Ensure values are Python float scalars, not numpy arrays
    if isinstance(brightness, np.ndarray):
        brightness = float(brightness.item() if brightness.size == 1 else brightness.mean())
    if isinstance(contrast, np.ndarray):
        contrast = float(contrast.item() if contrast.size == 1 else contrast.mean())
    if isinstance(timbre_complexity, np.ndarray):
        timbre_complexity = float(timbre_complexity.item() if timbre_complexity.size == 1 else timbre_complexity.mean())
    
    # Normalize values for radar chart
    brightness_norm = min(1.0, float(brightness))
    contrast_norm = min(1.0, float(contrast) / 50)  # Scale contrast to 0-1
    complexity_norm = min(1.0, float(timbre_complexity) / 40)  # Scale complexity to 0-1
    
    # Categories for radar chart
    categories = ['Brightness', 'Contrast', 'Complexity']
    values = [brightness_norm, contrast_norm, complexity_norm]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(67, 147, 195, 0.5)',
        line=dict(color='rgba(67, 147, 195, 1)'),
        customdata=["timbre"] * len(categories),  # Store feature name for component details
        hovertemplate='%{theta}: %{r:.2f}<extra></extra>'
    ))
    
    # Add annotations
    annotations = [
        dict(
            x=0.5,
            y=1.15,
            xref="paper",
            yref="paper",
            text=f"Brightness: {brightness_category}",
            showarrow=False,
            font=dict(size=14)
        ),
        dict(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"Contrast: {contrast_category}",
            showarrow=False,
            font=dict(size=14)
        ),
        dict(
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Complexity: {complexity_category}",
            showarrow=False,
            font=dict(size=14)
        )
    ]
    
    # Update layout
    fig.update_layout(
        title="Instrumentation Analysis",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        annotations=annotations,
        margin=dict(l=20, r=20, t=80, b=20),
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(245, 245, 245, 1)',
        paper_bgcolor='rgba(245, 245, 245, 1)'
    )
    
    return fig 