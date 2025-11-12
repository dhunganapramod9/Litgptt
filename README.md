# 💡 LitGPT Advanced Chat

An advanced, feature-rich Streamlit application for interacting with Large Language Models using LitGPT. This application provides a modern chat interface with streaming responses, conversation history, and extensive customization options.

## ✨ Features

### 🎯 Core Features
- **💬 Interactive Chat Interface**: Modern chat UI with conversation history
- **⚡ Streaming Responses**: Real-time token generation with live updates
- **🤖 Multiple Model Support**: Easy switching between different LLM models
- **📝 System Prompts**: Customize model behavior with system instructions
- **🔄 Conversation Management**: Save, export, and clear conversations

### 🎛️ Advanced Controls
- **Temperature Control**: Adjust creativity and randomness (0.0 - 2.0)
- **Top-K Sampling**: Limit token selection to top K candidates
- **Top-P (Nucleus) Sampling**: Dynamic token selection based on cumulative probability
- **Max Tokens**: Control response length (10 - 512 tokens)
- **Streaming Toggle**: Enable/disable real-time token streaming

### 📊 Analytics & Monitoring
- **Token Counting**: Track total tokens generated
- **Performance Metrics**: Monitor generation speed (tokens/second)
- **Generation Statistics**: Per-message and cumulative statistics
- **Model Information**: Display model parameters and device info

### 💾 Data Management
- **Export Conversations**: Download conversations as JSON
- **Clear History**: Reset conversation with one click
- **Model Reloading**: Hot-reload models without restarting

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd litchat
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download a model (choose one):**
   ```bash
   litgpt download --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0
   # or
   litgpt download --repo_id microsoft/phi-2
   # or
   litgpt download --repo_id meta-llama/Llama-2-7b-chat-hf
   ```

6. **Run the application:**
   ```bash
   streamlit run app.py
   ```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### Basic Usage

1. **Select a Model**: Choose from the dropdown in the sidebar
2. **Adjust Parameters**: Fine-tune generation settings as needed
3. **Type Your Message**: Enter your prompt in the chat input
4. **View Response**: Watch the AI generate text in real-time (if streaming is enabled)

### Advanced Features

#### System Prompts
Use system prompts to guide the model's behavior:
- `"You are a helpful coding assistant."` - For programming help
- `"You are a creative writing assistant."` - For creative content
- `"You are a scientific researcher."` - For technical explanations

#### Generation Parameters

- **Temperature** (0.0 - 2.0):
  - Lower (0.0-0.5): More deterministic, focused responses
  - Medium (0.6-0.9): Balanced creativity and coherence
  - Higher (1.0-2.0): More creative, diverse responses

- **Top-K** (0-100):
  - Limits sampling to the K most likely tokens
  - 0 = disabled (use all tokens)
  - Lower values = more focused, higher = more diverse

- **Top-P** (0.0 - 1.0):
  - Nucleus sampling threshold
  - 0.0 = only most likely token
  - 1.0 = all tokens considered
  - Works with Top-K for fine-grained control

#### Exporting Conversations

1. Click "📥 Export Conversation" in the sidebar
2. Click "Download JSON" to save the conversation
3. File includes messages, model used, and timestamp

## 🎨 Interface Overview

### Main Chat Area
- **Chat Messages**: User and assistant messages with timestamps
- **Statistics**: Per-message token count, generation time, and speed
- **Input Field**: Type your messages at the bottom

### Sidebar
- **Model Selection**: Choose and switch between models
- **Generation Parameters**: Fine-tune all sampling parameters
- **System Prompt**: Set behavioral instructions
- **Model Management**: Reload models, view statistics
- **Data Management**: Export and clear conversations

## 🛠️ Supported Models

The application supports any model compatible with LitGPT, including:

- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** - Fast, lightweight model
- **microsoft/phi-2** - High-quality small model
- **meta-llama/Llama-2-7b-chat-hf** - Powerful conversational model
- **mistralai/Mistral-7B-Instruct-v0.2** - Instruction-tuned model
- **HuggingFaceH4/zephyr-7b-beta** - Fine-tuned conversational model

To use a different model:
1. Download it: `litgpt download --repo_id <model_name>`
2. Add it to the model list in `app.py`
3. Select it from the dropdown

## 🔧 Configuration

### Customizing Models

Edit the `model_options` list in `app.py`:

```python
model_options = [
    "YourModel/ModelName",
    # Add more models here
]
```

### Adjusting Default Parameters

Modify the default values in the sidebar sliders:

```python
temperature = st.slider("Temperature", ..., value=0.7)  # Change default
max_tokens = st.slider("Max Tokens", ..., value=150)     # Change default
```

## 📊 Performance Tips

1. **Use GPU**: Significantly faster generation (automatic if CUDA available)
2. **Smaller Models**: Faster responses, lower memory usage
3. **Adjust Max Tokens**: Lower values = faster generation
4. **Disable Streaming**: Slightly faster for non-interactive use
5. **Model Caching**: Models are cached after first load

## 🐛 Troubleshooting

### Model Not Found
```
Error: Model not found
```
**Solution**: Download the model first:
```bash
litgpt download --repo_id <model_name>
```

### Out of Memory
```
CUDA out of memory
```
**Solution**: 
- Use a smaller model
- Reduce max_tokens
- Use CPU instead of GPU

### Slow Generation
**Solution**:
- Ensure GPU is being used
- Use a smaller model
- Reduce max_tokens
- Disable streaming

## License

This project uses LitGPT, which is licensed under the Apache License 2.0.

##  Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

##  Acknowledgments

- [LitGPT](https://github.com/Lightning-AI/litgpt) - Lightning AI's GPT implementation
- [Streamlit](https://streamlit.io/) - The web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework

## Support

For issues related to:
- **LitGPT**: Check the [LitGPT documentation](https://github.com/Lightning-AI/litgpt)
- **This Application**: Open an issue in the repository


