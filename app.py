# Enhanced Pneumonia Detection - Hugging Face Spaces Entry Point
# This file serves as the entry point for Hugging Face Spaces deployment

# Import the main Gradio interface
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

# Import and launch the Gradio interface
from gradio_ui import demo

# Launch the app
if __name__ == "__main__":
    demo.launch()