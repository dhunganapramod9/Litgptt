import streamlit as st
import time
from datetime import datetime
import json
import sys

# Try importing litgpt with error handling
try:
    from litgpt.api import LLM
    LITGPT_AVAILABLE = True
except ImportError as e:
    LITGPT_AVAILABLE = False
    st.error(f"❌ Failed to import litgpt: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="LitGPT Advanced Chat",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "current_model" not in st.session_state:
    st.session_state.current_model = ""
if "llm_instance" not in st.session_state:
    st.session_state.llm_instance = None
if "generation_stats" not in st.session_state:
    st.session_state.generation_stats = {"total_tokens": 0, "total_time": 0, "avg_speed": 0}

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_options = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "HuggingFaceH4/zephyr-7b-beta"
    ]
    
    selected_model = st.selectbox(
        "🤖 Model",
        model_options,
        index=0,
        help="Select the language model to use"
    )
    
    # Advanced generation parameters
    st.subheader("🎛️ Generation Parameters")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness. Lower = more deterministic, Higher = more creative"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=10,
        max_value=512,
        value=150,
        step=10,
        help="Maximum number of tokens to generate"
    )
    
    top_k = st.slider(
        "Top-K",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Consider only top K tokens. 0 = disabled"
    )
    
    top_p = st.slider(
        "Top-P (Nucleus)",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Nucleus sampling threshold"
    )
    
    use_streaming = st.checkbox(
        "Stream Response",
        value=True,
        help="Stream tokens as they're generated (real-time)"
    )
    
    # System prompt with templates
    st.subheader("📝 System Prompt")
    
    prompt_templates = {
        "Default": "You are a helpful AI assistant.",
        "Coding Assistant": "You are an expert programming assistant. Provide clear, well-commented code solutions.",
        "Creative Writer": "You are a creative writing assistant. Help craft engaging stories and narratives.",
        "Scientific Researcher": "You are a scientific researcher. Provide accurate, well-researched explanations.",
        "Teacher": "You are a patient teacher. Explain concepts clearly and adapt to different learning styles.",
        "Business Analyst": "You are a business analyst. Provide strategic insights and data-driven recommendations.",
        "Custom": ""
    }
    
    template_choice = st.selectbox(
        "Prompt Template",
        list(prompt_templates.keys()),
        help="Choose a preset or create custom instructions"
    )
    
    if template_choice == "Custom":
        system_prompt = st.text_area(
            "Custom System Instructions",
            value="",
            height=100,
            help="Enter your custom system prompt"
        )
    else:
        system_prompt = st.text_area(
            "System Instructions",
            value=prompt_templates[template_choice],
            height=100,
            help="Instructions to guide the model's behavior"
        )
    
    # Model management
    st.subheader("🔧 Model Management")
    if st.button("🔄 Reload Model"):
        st.session_state.model_loaded = False
        st.session_state.llm_instance = None
        st.session_state.current_model = None
        st.rerun()
    
    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Total Tokens", st.session_state.generation_stats["total_tokens"])
    st.metric("Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
    if st.session_state.generation_stats["total_time"] > 0:
        st.metric("Avg Speed", f"{st.session_state.generation_stats['avg_speed']:.1f} tokens/s")
        st.metric("Total Time", f"{st.session_state.generation_stats['total_time']:.1f}s")
    
    # Export/Import
    st.subheader("💾 Data Management")
    if st.button("📥 Export Conversation"):
        export_data = {
            "messages": st.session_state.messages,
            "model": selected_model,
            "timestamp": datetime.now().isoformat()
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.generation_stats = {"total_tokens": 0, "total_time": 0, "avg_speed": 0}
        st.rerun()

# Main content area
try:
    st.markdown('<div class="main-header">💡 LitGPT Advanced Chat</div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error rendering header: {e}")

# Model loading - completely lazy, only called when user sends message
def load_model(model_name):
    """Load and cache the model - only called on demand"""
    try:
        # Use session state to cache instead of @st.cache_resource to avoid startup issues
        cache_key = f"model_{model_name}"
        if cache_key in st.session_state and st.session_state[cache_key] is not None:
            return st.session_state[cache_key], None
        
        # Actually load the model
        llm = LLM.load(model_name)
        llm.distribute()
        
        # Cache in session state
        st.session_state[cache_key] = llm
        return llm, None
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg

# Lazy model loading - only load when needed
# Don't load on startup to avoid timeouts on Streamlit Cloud
if st.session_state.model_loaded and st.session_state.current_model == selected_model:
    # Model already loaded and matches selection
    pass
elif st.session_state.current_model != selected_model:
    # Model changed - reset state
    st.session_state.model_loaded = False
    st.session_state.llm_instance = None
    st.session_state.current_model = selected_model

# Show model status
if st.session_state.model_loaded:
    st.success(f"✅ Model ready: {selected_model.split('/')[-1]}")
else:
    st.info(f"💡 Model will load automatically when you send your first message. Selected: {selected_model.split('/')[-1]}")
    st.warning("⚠️ **Note:** On Streamlit Cloud, model loading may take 2-5 minutes on first use due to limited resources.")

# Display model info
try:
    if st.session_state.llm_instance:
        with st.expander("ℹ️ Model Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", selected_model.split("/")[-1])
            with col2:
                # Calculate parameters from the model
                try:
                    if hasattr(st.session_state.llm_instance, 'model'):
                        total_params = sum(p.numel() for p in st.session_state.llm_instance.model.parameters())
                        param_str = f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.2f}M"
                        st.metric("Parameters", param_str)
                    else:
                        st.metric("Parameters", "N/A")
                except Exception as e:
                    st.metric("Parameters", "N/A")
            with col3:
                try:
                    device = "CUDA" if hasattr(st.session_state.llm_instance, 'fabric') and st.session_state.llm_instance.fabric else "CPU"
                    st.metric("Device", device)
                except Exception:
                    st.metric("Device", "Unknown")
except Exception as e:
    st.warning(f"Could not display model info: {e}")

# Quick actions
st.subheader("⚡ Quick Actions")
quick_prompts = {
    "Explain": "Explain the concept of",
    "Write": "Write a short story about",
    "Code": "Write Python code to",
    "Summarize": "Summarize the following:",
    "Translate": "Translate to English:",
    "Brainstorm": "Brainstorm ideas for"
}

col1, col2, col3 = st.columns(3)
quick_cols = [col1, col2, col3]
for idx, (action, template) in enumerate(quick_prompts.items()):
    with quick_cols[idx % 3]:
        if st.button(f"📝 {action}", key=f"quick_{action}"):
            st.session_state.quick_prompt_template = template
            st.rerun()

# Chat interface
st.subheader("💬 Chat")

# Instructions for user
if len(st.session_state.messages) == 0:
    st.info("👆 **Type your message in the text box at the bottom of this page** to start chatting with the AI!")

# Display chat history
chat_container = st.container()
with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        timestamp = message.get("timestamp", "")
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
                if timestamp:
                    st.caption(f"🕐 {timestamp}")
        else:
            with st.chat_message("assistant"):
                st.write(content)
                if timestamp:
                    st.caption(f"🕐 {timestamp}")
                if "stats" in message:
                    stats = message["stats"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"📊 {stats.get('tokens', 0)} tokens")
                    with col2:
                        st.caption(f"⏱️ {stats.get('time', 0):.2f}s")
                    with col3:
                        if stats.get('time', 0) > 0:
                            st.caption(f"⚡ {stats.get('tokens', 0) / stats.get('time', 1):.1f} tok/s")

# Prompt input section - make it more visible
st.markdown("---")
st.markdown("### ✍️ Type Your Message")

# Alternative text area input (more visible)
use_text_area = st.checkbox("Use text area input (alternative to chat input)", value=False)

prompt = None  # Initialize prompt variable

if use_text_area:
    # Text area input
    if "quick_prompt_template" in st.session_state:
        user_input = st.text_area(
            f"💬 {st.session_state.quick_prompt_template}...",
            height=100,
            key="text_area_input"
        )
        if st.button("🚀 Send", type="primary", use_container_width=True):
            if user_input:
                prompt = st.session_state.quick_prompt_template + " " + user_input
                del st.session_state.quick_prompt_template
    else:
        user_input = st.text_area(
            "💬 Type your message here...",
            height=100,
            key="text_area_input"
        )
        if st.button("🚀 Send", type="primary", use_container_width=True):
            prompt = user_input if user_input else None
else:
    # Standard chat input (appears at bottom of page)
    st.caption("💡 **Tip:** The chat input box appears at the very bottom of this page. Scroll down to see it!")
    
    # Handle quick prompt
    if "quick_prompt_template" in st.session_state:
        chat_prompt = st.chat_input(f"{st.session_state.quick_prompt_template}...")
        if chat_prompt:
            prompt = st.session_state.quick_prompt_template + " " + chat_prompt
            del st.session_state.quick_prompt_template
    else:
        prompt = st.chat_input("💬 Type your message here and press Enter...")

if prompt:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Load model if not already loaded - ONLY when user sends message
            if not st.session_state.model_loaded or st.session_state.current_model != selected_model:
                with message_placeholder.container():
                    st.info(f"🔄 Loading {selected_model.split('/')[-1]}...")
                    st.caption("This may take 2-5 minutes on first use. Please be patient...")
                
                try:
                    llm, error = load_model(selected_model)
                    if error:
                        message_placeholder.error(f"❌ Error loading model: {str(error)[:500]}")
                        st.warning("💡 **Tip:** Streamlit Cloud has limited resources. Model loading may fail due to memory constraints.")
                        st.info("Consider using Hugging Face Spaces for better resource availability.")
                        st.stop()
                    else:
                        st.session_state.llm_instance = llm
                        st.session_state.model_loaded = True
                        st.session_state.current_model = selected_model
                        message_placeholder.empty()
                except MemoryError:
                    message_placeholder.error("❌ Out of memory! Streamlit Cloud doesn't have enough resources for this model.")
                    st.warning("💡 **Solution:** Deploy to Hugging Face Spaces or use a smaller model.")
                    st.stop()
                except Exception as e:
                    message_placeholder.error(f"❌ Unexpected error: {str(e)[:500]}")
                    st.stop()
            
            if not st.session_state.llm_instance:
                message_placeholder.error("❌ Model failed to load. Please try again or use a different model.")
                st.stop()
            
            start_time = time.time()
            tokens_generated = 0
            
            # Get accurate token count using tokenizer
            def count_tokens(text):
                if hasattr(st.session_state.llm_instance, 'tokenizer'):
                    try:
                        return len(st.session_state.llm_instance.tokenizer.encode(text))
                    except:
                        return len(text.split())  # Fallback to word count
                return len(text.split())  # Fallback to word count
            
            if use_streaming and st.session_state.llm_instance:
                # Streaming generation
                stream = st.session_state.llm_instance.generate(
                    prompt,
                    sys_prompt=system_prompt if system_prompt else None,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p,
                    stream=True
                )
                
                for chunk in stream:
                    full_response += chunk
                    message_placeholder.write(full_response + "▌")
                
                message_placeholder.write(full_response)
                tokens_generated = count_tokens(full_response) - count_tokens(prompt)
            else:
                # Non-streaming generation
                with st.spinner("⚙️ Generating..."):
                    full_response = st.session_state.llm_instance.generate(
                        prompt,
                        sys_prompt=system_prompt if system_prompt else None,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k if top_k > 0 else None,
                        top_p=top_p,
                        stream=False
                    )
                    tokens_generated = count_tokens(full_response) - count_tokens(prompt)
                    message_placeholder.write(full_response)
            
            elapsed_time = time.time() - start_time
            
            # Update statistics
            st.session_state.generation_stats["total_tokens"] += tokens_generated
            st.session_state.generation_stats["total_time"] += elapsed_time
            if st.session_state.generation_stats["total_time"] > 0:
                st.session_state.generation_stats["avg_speed"] = (
                    st.session_state.generation_stats["total_tokens"] / 
                    st.session_state.generation_stats["total_time"]
                )
            
            # Display stats
            stats = {
                "tokens": tokens_generated,
                "time": elapsed_time
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📊 {tokens_generated} tokens")
            with col2:
                st.caption(f"⏱️ {elapsed_time:.2f}s")
            with col3:
                if elapsed_time > 0:
                    st.caption(f"⚡ {tokens_generated / elapsed_time:.1f} tok/s")
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stats": stats
            })
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    st.rerun()

# Footer
st.markdown("---")
st.caption("💡 Powered by LitGPT | Advanced AI Text Generation")
