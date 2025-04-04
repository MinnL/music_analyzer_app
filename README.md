# Real-Time Music Analysis & Visualization Tool

A web application that captures real-time audio input, analyzes and classifies music genres, and provides interactive visualizations of musical components such as rhythm, melody, and instrumentation.

## Features

- **Real-Time Audio Input**: Capture audio from your microphone with low latency
- **Music Genre Classification**: Identify music genres using audio analysis
- **Musical Component Analysis**: Decompose audio into rhythm, melody, and instrumentation components
- **Interactive Visualizations**: View real-time, interactive visualizations of musical features
- **Detailed Component Descriptions**: Explore in-depth information about musical elements

## Installation

1. Clone this repository:
   ```
   git clone [repository-url]
   cd music_analyzer_app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: You may need to install PortAudio first for PyAudio to work properly.
   
   - On macOS: `brew install portaudio`
   - On Linux: `sudo apt-get install portaudio19-dev`
   - On Windows: PyAudio wheel should handle this automatically

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8050
   ```

3. Click the "Start Recording" button to begin capturing audio from your microphone.

4. Play or perform music, and watch as the application:
   - Classifies the genre of the music
   - Displays real-time visualizations of the rhythm, melody, and instrumentation
   - Updates the analysis as the music changes

5. Click on different parts of the visualizations to view detailed information about specific musical components.

6. Click "Stop Recording" when finished.

## System Requirements

- Python 3.8 or higher
- A working microphone
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Technical Architecture

The application consists of several key components:

1. **Audio Input Module**: Captures real-time audio from the microphone using PyAudio
2. **Analysis Module**: Processes audio data, extracts features, and classifies genres using librosa and machine learning models
3. **Visualization Module**: Creates interactive visualizations using Plotly
4. **Web Interface**: Built with Dash, providing a responsive and interactive user experience

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The librosa team for their audio and music analysis tools
- The Dash and Plotly teams for their data visualization libraries 