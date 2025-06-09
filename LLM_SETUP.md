# LLM Integration Setup for Genra

## Overview

Genra now supports dynamic genre explanations using Large Language Models (LLMs) instead of hardcoded explanations. This provides more engaging, contextual, and educational explanations for genre classifications.

## Supported LLM Providers

Currently supported:
- **OpenAI GPT-3.5-turbo** (Primary)
- **Fallback to hardcoded explanations** (Always available)

Future support planned for:
- Anthropic Claude
- Local models (Ollama, etc.)
- Google Gemini

## Setup Instructions

### 1. Install Dependencies

The OpenAI library is already included in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

#### Option A: Environment Variable (Recommended)
```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### Option B: Create .env file (in project root)
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 3. Get OpenAI API Key

1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Create a new API key
4. Copy the key and set it as shown above

### 4. Configuration Options

You can configure LLM behavior in `app/analysis.py`:

```python
# In MusicAnalyzer.__init__()
self.use_llm_explanations = True  # Set to False to disable LLMs
self.llm_provider = "openai"      # Current provider
```

## How It Works

### LLM Explanation Flow

1. **Feature Extraction**: The system extracts key audio features (tempo, instruments, etc.)
2. **Prompt Generation**: Creates a detailed prompt asking the LLM to explain the genre classification
3. **LLM Processing**: Sends the prompt to OpenAI GPT-3.5-turbo
4. **Response Formatting**: Converts the LLM response into formatted HTML for display
5. **Fallback**: If LLM fails, automatically falls back to hardcoded explanations

### Example LLM Prompt

```
I need you to explain why a piece of audio was classified as ROCK music based on the following analyzed features:

- Tempo: 120.5 BPM
- Rhythm complexity: Medium
- Key/Mode: C Major
- Pitch variety: High
- Timbral brightness: Bright/Sharp
- Detected instruments: Electric Guitar, Drums, Bass

Please provide a clear, educational explanation that covers:
1. A brief introduction to what rock music is
2. How the detected features match typical rock characteristics
3. Specific connections between the audio features and the genre
4. Any interesting musical insights about why these features are characteristic of rock

Keep the explanation engaging but informative, suitable for both music enthusiasts and casual listeners.
```

## Benefits of LLM Explanations

### Compared to Hardcoded Explanations:

**LLM Advantages:**
- **Dynamic Content**: Each explanation is unique based on specific audio features
- **Educational Depth**: More detailed and contextual explanations
- **Natural Language**: Conversational, engaging tone
- **Adaptive**: Can adjust explanations based on feature combinations
- **Current Knowledge**: Benefits from LLM's training on music theory and genres

**Hardcoded Advantages:**
- **Fast**: No API calls required
- **Reliable**: Always available, no network dependency
- **Cost**: No per-use fees
- **Consistent**: Same explanation for same feature patterns

## Cost Considerations

- **OpenAI GPT-3.5-turbo**: ~$0.001-0.002 per explanation
- **Monthly usage**: For typical usage (~100-500 explanations), cost is $0.10-1.00
- **Cost control**: Set usage limits in OpenAI dashboard

## Troubleshooting

### LLM Not Working
1. **Check API Key**: Ensure `OPENAI_API_KEY` is set correctly
2. **Check Internet**: LLM requires internet connection
3. **Check Logs**: Look for error messages in console
4. **Fallback**: App automatically falls back to hardcoded explanations

### Common Issues

**"WARNING: OPENAI_API_KEY not found"**
- Solution: Set the environment variable as shown above

**"LLM explanation failed: ... Falling back to hardcoded explanations"**
- This is normal behavior when LLM is unavailable
- Check your API key and internet connection

**Slow explanations**
- Normal: LLM calls take 1-3 seconds
- If consistently slow, check OpenAI API status

## Future Enhancements

### Planned Features:
1. **Multiple LLM Providers**: Support for Anthropic Claude, local models
2. **Explanation Styles**: Choose between technical, casual, or educational explanations
3. **Caching**: Cache LLM responses for repeated feature combinations
4. **Streaming**: Real-time explanation generation for longer explanations
5. **Custom Prompts**: User-configurable explanation styles

### Integration Ideas:
- **Instrument Explanations**: Use LLMs for instrument detection explanations
- **Genre Education**: Generate comprehensive genre knowledge sections
- **Comparative Analysis**: "This sounds like rock but has jazz influences because..."
- **User Questions**: Allow users to ask follow-up questions about the analysis

## Development Notes

### Adding New LLM Providers

To add a new LLM provider, modify the `_get_llm_genre_explanation` method:

```python
elif self.llm_provider == "anthropic":
    # Add Anthropic Claude integration
    pass
elif self.llm_provider == "local":
    # Add local model integration (Ollama, etc.)
    pass
```

### Customizing Prompts

Modify `_create_explanation_prompt()` to change how explanations are generated:

```python
def _create_explanation_prompt(self, genre, features_summary, components):
    # Customize prompt structure here
    prompt = f"Custom prompt for {genre}..."
    return prompt
```

## Security Notes

- **API Key Security**: Never commit API keys to version control
- **Environment Variables**: Use environment variables for production
- **Rate Limiting**: OpenAI has built-in rate limiting
- **Input Validation**: All user inputs are validated before sending to LLM

---

## Quick Start

1. Get OpenAI API key
2. Set environment variable: `export OPENAI_API_KEY="your-key"`
3. Run Genra: `python app.py`
4. Upload/record audio and see LLM-powered explanations!

The system will automatically detect if LLM is available and fall back gracefully if not. 