"""
Chest X-ray Pneumonia Detection Explainability Package

This package provides comprehensive explainability tools for TensorFlow CNN models
including SHAP, LIME, Grad-CAM, and automated reporting capabilities.
"""

__version__ = "1.0.0"

# Import main functions if dependencies are available
try:
    from .pipeline import generate_complete_explanation, batch_explain, ExplanationArtifacts
    __all__ = ["generate_complete_explanation", "batch_explain", "ExplanationArtifacts"]
except ImportError as e:
    print(f"Warning: Some explainability features may not be available due to missing dependencies: {e}")
    __all__ = []