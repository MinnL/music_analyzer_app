# Real-Time Music Analysis & Visualization Tool

A web application that captures real-time audio input, analyzes and classifies music genres, and provides interactive visualizations of musical components such as rhythm, melody, and instrumentation.

## Features

- **Real-Time Audio Input**: Capture audio from your microphone with low latency
- **Audio File Upload**: Upload and analyze your own audio files in various formats (WAV, MP3, OGG, FLAC, M4A)
- **Audio Playback**: Replay recorded sounds or uploaded music files through the web interface
- **Music Genre Classification**: Identify music genres using audio analysis
- **Musical Component Analysis**: Decompose audio into rhythm, melody, and instrumentation components
- **Interactive Visualizations**: View real-time, interactive visualizations of musical features
- **Detailed Component Descriptions**: Explore in-depth information about musical elements
- **Demo Mode**: If microphone access fails, the app automatically switches to a demo mode

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/MinnL/music_analyzer_app.git
   cd music_analyzer_app
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv music_analyzer_env
   source music_analyzer_env/bin/activate  # On Windows: music_analyzer_env\Scripts\activate
   ```

3. Install the required dependencies:
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
   http://127.0.0.1:8050
   ```

3. You have two options for analyzing music:

   **Option 1: Microphone Input**
   - Click the "Start Recording" button to begin capturing audio from your microphone
   - Play or perform music, and watch the real-time analysis
   - Click "Stop Recording" when finished

   **Option 2: Upload an Audio File**
   - In the "Audio Input Controls" section, use the upload area to drag and drop an audio file or click to select one
   - Supported formats: WAV, MP3, OGG, FLAC, M4A
   - The application will automatically analyze the uploaded file and display the results

4. View the analysis results:
   - See the detected genre at the top of the page
   - Explore the rhythm, melody, and instrumentation visualizations
   - Click on different elements of the visualizations to see detailed information about specific musical components

5. Play back the audio:
   - After recording or uploading a file, the "Play Audio" button will become active
   - Click the button to play back the most recent audio
   - For recordings, this captures what was recorded from your microphone or the demo audio
   - For file uploads, this plays back the uploaded file
   - The audio player provides standard controls (play/pause, timeline scrubbing, volume)

6. If you encounter issues with microphone access (common on macOS), the app will automatically switch to demo mode, allowing you to still test the visualizations.

## Troubleshooting

### Microphone Access Issues
- On macOS, you may see PortAudio errors. This is a known issue with PyAudio on Mac systems.
- The app will automatically fall back to demo mode, which simulates audio input.
- You can still use the file upload feature to analyze your own music files.

### File Upload Problems
- Ensure your audio file is in one of the supported formats (WAV, MP3, OGG, FLAC, M4A)
- Large files may take longer to process
- If the app becomes unresponsive, try refreshing the page

### Audio Playback Issues
- If playback doesn't work, ensure your browser supports HTML5 audio
- Try different audio file formats if you encounter compatibility issues
- For very large files, there might be a delay before playback starts

## System Requirements

- Python 3.8 or higher
- A working microphone (optional, as you can use file upload)
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Technical Architecture

The application consists of several key components:

1. **Audio Input Module**: Captures real-time audio from the microphone using PyAudio or processes uploaded audio files
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