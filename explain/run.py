"""
Command Line Interface for Chest X-ray Pneumonia Detection Explainability.

Provides CLI access to the explainability pipeline with support for
single image processing and batch processing from CSV files.
"""

import argparse
import sys
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import numpy as np
    import gradio as gr
    import tempfile
    
    from explain.pipeline import generate_complete_explanation, batch_explain
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)


def load_pneumonia_model():
    """
    Load the pre-trained pneumonia detection model.
    
    Returns:
        TensorFlow/Keras model
    """
    print("Loading pneumonia detection model...")
    try:
        model_path = hf_hub_download(
            repo_id="ayushirathour/chest-xray-pneumonia-detection",
            filename="best_chest_xray_model.h5"
        )
        model = tf.keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)


def create_preprocess_function():
    """
    Create the preprocessing function used by the original model.
    
    Returns:
        Preprocessing function
    """
    def preprocess_xray_pil(pil_img: Image.Image) -> np.ndarray:
        img = pil_img.convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    
    return preprocess_xray_pil


def process_single_image(args):
    """
    Process a single chest X-ray image.
    
    Args:
        args: Parsed command line arguments
    """
    print(f"Processing single image: {args.image}")
    
    # Validate input
    if not os.path.exists(args.image):
        print(f"✗ Image file not found: {args.image}")
        sys.exit(1)
    
    # Load model and preprocessing function
    model = load_pneumonia_model()
    preprocess_fn = create_preprocess_function()
    
    # Set up class names
    class_names = ['Normal', 'Pneumonia']
    
    # Parse metadata if provided
    metadata = {}
    if args.metadata:
        for item in args.metadata:
            if '=' in item:
                key, value = item.split('=', 1)
                metadata[key.strip()] = value.strip()
    
    # Generate explanation
    try:
        artifacts = generate_complete_explanation(
            model=model,
            image_path=args.image,
            patient_id=args.patient_id,
            class_names=class_names,
            output_dir=args.output_dir,
            preprocess_fn=preprocess_fn,
            metadata=metadata
        )
        
        print("\\n✓ Explanation generation completed successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📄 HTML Report: {artifacts.html_report}")
        print(f"🔥 Grad-CAM Heatmap: {artifacts.gradcam_heatmap}")
        print(f"🔍 LIME Positive: {artifacts.lime_positive}")
        print(f"📈 Probability Chart: {artifacts.probability_chart}")
        
        if artifacts.shap_waterfall:
            print(f"📊 SHAP Waterfall: {artifacts.shap_waterfall}")
        
    except Exception as e:
        print(f"✗ Failed to generate explanations: {e}")
        sys.exit(1)


def process_batch_csv(args):
    """
    Process multiple images from a CSV file.
    
    Args:
        args: Parsed command line arguments
    """
    print(f"Processing batch from CSV: {args.batch_csv}")
    
    # Validate CSV file
    if not os.path.exists(args.batch_csv):
        print(f"✗ CSV file not found: {args.batch_csv}")
        sys.exit(1)
    
    # Load CSV
    try:
        df = pd.read_csv(args.batch_csv)
        print(f"✓ Loaded CSV with {len(df)} entries")
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_columns = ['image_path', 'patient_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"✗ Missing required columns in CSV: {missing_columns}")
        print(f"Required columns: {required_columns}")
        sys.exit(1)
    
    # Load model and preprocessing function
    model = load_pneumonia_model()
    preprocess_fn = create_preprocess_function()
    
    # Prepare dataset
    dataset = []
    for _, row in df.iterrows():
        item = {
            'image_path': row['image_path'],
            'patient_id': row['patient_id'],
            'metadata': {}
        }
        
        # Add any additional columns as metadata
        for col in df.columns:
            if col not in required_columns:
                item['metadata'][col] = row[col]
        
        dataset.append(item)
    
    # Process batch
    try:
        results_df = batch_explain(
            model=model,
            dataset=dataset,
            output_dir=args.output_dir,
            class_names=['Normal', 'Pneumonia'],
            preprocess_fn=preprocess_fn
        )
        
        print("\\n✓ Batch processing completed successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📋 Results manifest: {os.path.join(args.output_dir, 'batch_results_manifest.csv')}")
        print(f"📊 Processed {len(results_df)} patients")
        
        # Show success/failure summary
        successful = len(results_df[~results_df.get('error', pd.Series()).notna()])
        failed = len(results_df) - successful
        print(f"✓ Successful: {successful}")
        if failed > 0:
            print(f"✗ Failed: {failed}")
        
    except Exception as e:
        print(f"✗ Failed to process batch: {e}")
        sys.exit(1)


def create_sample_csv(output_path: str):
    """
    Create a sample CSV file for batch processing.
    
    Args:
        output_path: Path to save the sample CSV
    """
    sample_data = {
        'patient_id': ['P001', 'P002', 'P003'],
        'image_path': [
            '/path/to/chest_xray_001.jpg',
            '/path/to/chest_xray_002.jpg', 
            '/path/to/chest_xray_003.jpg'
        ],
        'age': [45, 62, 38],
        'gender': ['M', 'F', 'M'],
        'symptoms': ['cough, fever', 'shortness of breath', 'chest pain']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"Sample CSV created: {output_path}")
    print("Edit this file with your actual image paths and patient data.")


def launch_gradio_interface():
    """
    Launch the interactive Gradio web interface.
    """
    print("🚀 Launching Gradio Web Interface...")
    
    # Load model once
    model = load_pneumonia_model()
    
    def process_image(image, patient_id, age, temperature, wbc_count, o2_sat, chest_pain):
        """Process uploaded image and return enhanced explanations"""
        try:
            # Save uploaded image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                image.save(tmp_file.name, 'JPEG')
                temp_image_path = tmp_file.name
            
            # Create temporary output directory
            temp_output_dir = tempfile.mkdtemp()
            
            # Prepare metadata
            metadata = {
                'age': age or '45 years',
                'temperature': temperature or '37.2°C',
                'wbc_count': wbc_count or '8,500',
                'o2_saturation': o2_sat or '96%',
                'chest_pain': chest_pain or 'No'
            }
            
            # Generate explanations
            artifacts = generate_complete_explanation(
                model=model,
                image_path=temp_image_path,
                patient_id=patient_id or 'WEB_USER',
                class_names=['Normal', 'Pneumonia'],
                output_dir=temp_output_dir,
                preprocess_fn=lambda x: tf.cast(tf.image.resize(x, [224, 224]), tf.float32) / 255.0,
                metadata=metadata
            )
            
            # Clean up temp image
            os.unlink(temp_image_path)
            
            # Return paths to generated images
            results = []
            if artifacts.probability_chart and os.path.exists(artifacts.probability_chart):
                results.append(artifacts.probability_chart)
            if artifacts.gradcam_heatmap and os.path.exists(artifacts.gradcam_heatmap):
                results.append(artifacts.gradcam_heatmap)
            if artifacts.lime_positive and os.path.exists(artifacts.lime_positive):
                results.append(artifacts.lime_positive)
            if artifacts.shap_waterfall and os.path.exists(artifacts.shap_waterfall):
                results.append(artifacts.shap_waterfall)
            
            return results
            
        except Exception as e:
            return [f"Error processing image: {str(e)}"]
    
    # Create Gradio interface
    with gr.Blocks(title="🏥 Enhanced Pneumonia Detection Explainability", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🏥 Enhanced Pneumonia Detection Explainability System</h1>
            <p>Upload a chest X-ray image to get AI explanations matching clinical target visualizations</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📋 Input</h3>")
                image_input = gr.Image(type="pil", label="Upload Chest X-Ray")
                
                with gr.Accordion("Patient Information (Optional)", open=False):
                    patient_id = gr.Textbox(label="Patient ID", placeholder="e.g., PAT001")
                    age = gr.Textbox(label="Age", placeholder="e.g., 45 years")
                    temperature = gr.Textbox(label="Temperature", placeholder="e.g., 38.5°C")
                    wbc_count = gr.Textbox(label="WBC Count", placeholder="e.g., 12,500")
                    o2_sat = gr.Textbox(label="O2 Saturation", placeholder="e.g., 92%")
                    chest_pain = gr.Textbox(label="Chest Pain", placeholder="e.g., Yes/No")
                
                analyze_btn = gr.Button("🔬 Analyze X-Ray", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.HTML("<h3>📊 Enhanced Explanations</h3>")
                output_gallery = gr.Gallery(
                    label="Target-Style Visualizations",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto"
                )
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: #f0f8ff; border-radius: 10px; margin: 20px 0;">
            <h4>🎯 Generated Visualizations Match Your Target Images:</h4>
            <p>✓ <b>Prediction Dashboard:</b> Clinical interface with patient vitals<br>
               ✓ <b>Grad-CAM Heatmap:</b> Red attention maps on X-ray regions<br>
               ✓ <b>LIME Explanation:</b> Superpixel analysis with feature importance<br>
               ✓ <b>SHAP Analysis:</b> Medical feature importance with blue-red gradient</p>
        </div>
        """)
        
        # Connect the button to the processing function
        analyze_btn.click(
            fn=process_image,
            inputs=[image_input, patient_id, age, temperature, wbc_count, o2_sat, chest_pain],
            outputs=[output_gallery]
        )
    
    # Launch the interface
    print("🌐 Starting web interface...")
    print("📱 The interface will open in your default web browser")
    print("🔄 Processing may take a few moments for each analysis")
    
    demo.launch(
        inbrowser=True,
        share=False,
        show_error=True,
        quiet=False
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chest X-ray Pneumonia Detection Explainability CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python -m explain.run --image chest_xray.jpg --patient-id P001 --output-dir ./results
  
  # Process single image with metadata
  python -m explain.run --image chest_xray.jpg --patient-id P001 --output-dir ./results \\
    --metadata age=45 gender=M symptoms="cough,fever"
  
  # Process batch from CSV
  python -m explain.run --batch-csv patients.csv --output-dir ./batch_results
  
  # Create sample CSV template
  python -m explain.run --create-sample-csv sample_patients.csv
        """
    )
    
    # Create mutually exclusive group for operation mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    mode_group.add_argument(
        '--image', 
        type=str,
        help='Path to chest X-ray image for single processing'
    )
    
    mode_group.add_argument(
        '--batch-csv',
        type=str,
        help='Path to CSV file with batch processing data'
    )
    
    mode_group.add_argument(
        '--create-sample-csv',
        type=str,
        help='Create a sample CSV template at the specified path'
    )
    
    mode_group.add_argument(
        '--gradio',
        action='store_true',
        help='Launch interactive web interface using Gradio'
    )
    
    # Common arguments
    parser.add_argument(
        '--patient-id',
        type=str,
        default='PATIENT_001',
        help='Patient identifier (required for single image processing)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./explanation_results',
        help='Output directory for generated artifacts (default: ./explanation_results)'
    )
    
    parser.add_argument(
        '--metadata',
        nargs='*',
        help='Metadata in key=value format (e.g., age=45 gender=M)'
    )
    
    parser.add_argument(
        '--disable-shap',
        action='store_true',
        help='Disable SHAP explanations (faster processing)'
    )
    
    parser.add_argument(
        '--lime-samples',
        type=int,
        default=1000,
        help='Number of samples for LIME explanation (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.create_sample_csv:
        create_sample_csv(args.create_sample_csv)
    elif args.gradio:
        launch_gradio_interface()
    elif args.image:
        if not args.patient_id:
            print("✗ Patient ID is required for single image processing")
            sys.exit(1)
        process_single_image(args)
    elif args.batch_csv:
        process_batch_csv(args)


if __name__ == '__main__':
    main()