from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
from PIL import Image
import gradio as gr
import os
import sys
from datetime import datetime

# Add parent directory to path for explain imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from explain.pipeline import generate_complete_explanation, ExplanationArtifacts
    EXPLAIN_AVAILABLE = True
except ImportError:
    print("Explainability features not available. Install requirements: pip install -r requirements.txt")
    EXPLAIN_AVAILABLE = False

# Download and load model (from the Hugging Face repo)
model_path = hf_hub_download(
    repo_id="ayushirathour/chest-xray-pneumonia-detection",
    filename="best_chest_xray_model.h5"
)
model = tf.keras.models.load_model(model_path)

# Preprocess as per the repo README: RGB, 224x224, scaled to [0,1]
def preprocess_xray_pil(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Gradio predict function
def predict_fn(image):
    # Gradio provides a numpy array (H, W, C) uint8 when type="numpy"
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image.astype("uint8"), "RGB")
    else:
        pil = Image.open(image).convert("RGB")
    x = preprocess_xray_pil(pil)
    prob = float(model.predict(x, verbose=0)[0][0])  # probability of pneumonia
    if prob >= 0.5:
        diagnosis = "PNEUMONIA"
        confidence = prob * 100
    else:
        diagnosis = "NORMAL"
        confidence = (1 - prob) * 100
    return {
        "Diagnosis": diagnosis,
        "Confidence (%)": round(confidence, 1),
        "Raw probability (pneumonia)": round(prob, 4)
    }


# Enhanced predict function with explanations
def predict_with_explanations(image, enable_explanations=True):
    """
    Predict with optional explainability analysis.
    
    Args:
        image: Input chest X-ray image
        enable_explanations: Whether to generate explanation artifacts
        
    Returns:
        Tuple of (prediction_dict, explanation_report_path or None)
    """
    # Get basic prediction
    prediction = predict_fn(image)
    
    if not enable_explanations or not EXPLAIN_AVAILABLE:
        return prediction, None
    
    try:
        # Save temporary image for explainability pipeline
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_id = f"TEMP_{timestamp}"
        temp_dir = "./temp_explanations"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Convert image to PIL and save temporarily
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype("uint8"), "RGB")
        else:
            pil_img = Image.open(image).convert("RGB")
        
        temp_image_path = os.path.join(temp_dir, f"{patient_id}.png")
        pil_img.save(temp_image_path)
        
        # Generate complete explanation
        output_dir = os.path.join(temp_dir, patient_id)
        
        artifacts = generate_complete_explanation(
            model=model,
            image_path=temp_image_path,
            patient_id=patient_id,
            class_names=['Normal', 'Pneumonia'],
            output_dir=output_dir,
            preprocess_fn=preprocess_xray_pil,
            metadata={
                'timestamp': timestamp,
                'prediction': prediction['Diagnosis'],
                'confidence': prediction['Confidence (%)']
            }
        )
        
        # Clean up temporary image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return prediction, artifacts.html_report
        
    except Exception as e:
        print(f"Explanation generation failed: {e}")
        return prediction, None

# Enhanced Gradio interface with explanations
def gradio_predict_with_explanations(image, enable_explanations):
    """Gradio wrapper for prediction with explanations."""
    prediction, report_path = predict_with_explanations(image, enable_explanations)
    
    if report_path and os.path.exists(report_path):
        # Return prediction and path to HTML report
        return prediction, f"📄 Explanation report generated: {report_path}"
    else:
        return prediction, "ℹ️ Explanations disabled or unavailable"


# Create enhanced interface
with gr.Blocks(title="Chest X-ray Pneumonia Detection with Explainability") as demo:
    gr.Markdown("""
    # 🫁 Chest X-ray Pneumonia Detection with AI Explainability
    
    Upload a chest X-ray image to get an AI-powered pneumonia diagnosis. 
    Enable explanations to generate comprehensive analysis including:
    - 🔥 **Grad-CAM**: Visual attention heatmaps
    - 🔍 **LIME**: Superpixel-level explanations  
    - 📊 **SHAP**: Feature importance analysis
    - 📈 **Probability Dashboard**: Confidence metrics
    - 📄 **HTML Report**: Complete analysis report
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="numpy", 
                label="📤 Upload Chest X-ray Image"
            )
            
            enable_explanations = gr.Checkbox(
                label="🔬 Enable AI Explainability Analysis",
                value=EXPLAIN_AVAILABLE,
                interactive=EXPLAIN_AVAILABLE,
                info="Generate detailed explanations (takes longer but provides insights)"
            )
            
            predict_btn = gr.Button(
                "🔍 Analyze X-ray", 
                variant="primary"
            )
        
        with gr.Column():
            prediction_output = gr.JSON(
                label="🎯 Prediction Results"
            )
            
            explanation_output = gr.Textbox(
                label="📋 Explanation Status",
                placeholder="Explanation status will appear here..."
            )
    
    # Add examples
    gr.Examples(
        examples=[],  # Add paths to example images if available
        inputs=image_input,
        label="📚 Example X-ray Images"
    )
    
    # Event handlers
    predict_btn.click(
        fn=gradio_predict_with_explanations,
        inputs=[image_input, enable_explanations],
        outputs=[prediction_output, explanation_output]
    )
    
    # Add information about explanations
    if not EXPLAIN_AVAILABLE:
        gr.Markdown("""
        ⚠️ **Explainability features not available**
        
        To enable AI explanations, install the required packages:
        ```bash
        pip install -r requirements.txt
        ```
        """)
    else:
        gr.Markdown("""
        ✅ **Explainability features enabled**
        
        When explanations are enabled, the system generates:
        - Grad-CAM attention heatmaps showing model focus areas
        - LIME superpixel analysis of positive/negative regions
        - SHAP feature importance waterfall charts
        - Comprehensive HTML reports with all visualizations
        """)

demo.launch(share=False)  # In notebooks, this displays inline
