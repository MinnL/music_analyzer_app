// main.js - Client-side functionality for the Music Analyzer app

document.addEventListener('DOMContentLoaded', function() {
    console.log('Music Analyzer App initialized');
    
    // This file can be used for any additional client-side interactivity
    // that's not already handled by the Dash framework
    
    // For example, we could add custom animation effects,
    // local audio processing, or enhanced UI interactions
    
    // Listen for messages from the Dash app
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'musicAnalyzerEvent') {
            handleCustomEvent(event.data);
        }
    });
    
    function handleCustomEvent(data) {
        // Handle custom events from the Dash app
        switch(data.action) {
            case 'recordingStarted':
                addVisualFeedback('recording');
                break;
            case 'recordingStopped':
                removeVisualFeedback('recording');
                break;
            case 'genreDetected':
                highlightGenre(data.genre);
                break;
            default:
                console.log('Unknown event:', data);
        }
    }
    
    function addVisualFeedback(type) {
        if (type === 'recording') {
            // Add a recording indicator or animation
            const body = document.body;
            const indicator = document.createElement('div');
            indicator.id = 'recording-indicator';
            indicator.innerHTML = '<div class="pulse"></div>';
            body.appendChild(indicator);
        }
    }
    
    function removeVisualFeedback(type) {
        if (type === 'recording') {
            // Remove recording indicator
            const indicator = document.getElementById('recording-indicator');
            if (indicator) {
                indicator.remove();
            }
        }
    }
    
    function highlightGenre(genre) {
        // Add visual highlight to the detected genre
        console.log('Genre detected:', genre);
        
        // This could be used to add special effects or animations
        // when a genre is detected
    }
}); 