"""
Main Pipeline for Chest X-ray Pneumonia Detection Explainability.

Provides the complete pipeline function that generates all explanation artifacts
with a single call, supports batch processing, and integrates all explainability methods.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

from .gradcam import generate_gradcam_explanation
from .lime_explain import generate_lime_explanation
from .shap_explain import generate_single_shap_explanation
from .dashboard import generate_probability_dashboard
from .report import generate_html_report


@dataclass
class ExplanationArtifacts:
    """
    Data class to hold paths to all generated explanation artifacts.
    """
    gradcam_heatmap: str
    gradcam_overlay: str
    lime_positive: str
    lime_negative: str
    shap_waterfall: Optional[str]
    probability_chart: str
    html_report: str
    patient_id: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for easy access."""
        return {
            'gradcam_heatmap': self.gradcam_heatmap,
            'gradcam_overlay': self.gradcam_overlay,
            'lime_positive': self.lime_positive,
            'lime_negative': self.lime_negative,
            'shap_waterfall': self.shap_waterfall,
            'probability_chart': self.probability_chart,
            'html_report': self.html_report
        }


class ExplainabilityPipeline:
    """
    Complete explainability pipeline for chest X-ray pneumonia detection.
    
    Integrates Grad-CAM, LIME, SHAP, and probability visualizations into
    a unified pipeline with automated report generation.
    """
    
    def __init__(self, model: tf.keras.Model, preprocess_fn: Callable,
                 class_names: List[str] = None):
        """
        Initialize the explainability pipeline.
        
        Args:
            model: TensorFlow/Keras model for pneumonia detection
            preprocess_fn: Function to preprocess images for the model
            class_names: List of class names (default: ['Normal', 'Pneumonia'])
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.class_names = class_names or ['Normal', 'Pneumonia']
    
    def _load_and_prepare_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare image for processing.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (original_image_array, preprocessed_image_array)
        """
        # Load original image
        pil_img = Image.open(image_path).convert('RGB')
        original_img = np.array(pil_img)
        
        # Preprocess for model
        preprocessed = self.preprocess_fn(pil_img)
        
        return original_img, preprocessed
    
    def _get_model_prediction(self, preprocessed_img: np.ndarray) -> Dict[str, float]:
        """
        Get model prediction and format results.
        
        Args:
            preprocessed_img: Preprocessed image array
            
        Returns:
            Dictionary containing prediction results
        """
        prediction = self.model.predict(preprocessed_img, verbose=0)
        
        if len(prediction.shape) == 2 and prediction.shape[1] == 1:
            # Binary output (pneumonia probability)
            pneumonia_prob = float(prediction[0][0])
            normal_prob = 1 - pneumonia_prob
            probabilities = np.array([normal_prob, pneumonia_prob])
        else:
            # Multi-class output
            probabilities = prediction[0]
            pneumonia_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        # Determine diagnosis
        if pneumonia_prob >= 0.5:
            diagnosis = "PNEUMONIA"
            confidence = pneumonia_prob * 100
        else:
            diagnosis = "NORMAL"
            confidence = (1 - pneumonia_prob) * 100
        
        return {
            'diagnosis': diagnosis,
            'confidence': confidence,
            'raw_probability': pneumonia_prob,
            'probabilities': probabilities
        }
    
    def generate_complete_explanation(self, image_path: str, patient_id: str,
                                    output_dir: str, metadata: Dict = None,
                                    enable_shap: bool = True,
                                    lime_samples: int = 1000) -> ExplanationArtifacts:
        """
        Generate complete explanation for a single chest X-ray image.
        
        This is the main function that generates all explanation artifacts:
        - Grad-CAM heatmaps and overlays
        - LIME positive/negative superpixel explanations  
        - SHAP waterfall plots
        - Probability dashboard
        - Comprehensive HTML report
        
        Args:
            image_path: Path to the chest X-ray image
            patient_id: Unique identifier for the patient
            output_dir: Directory to save all generated artifacts
            metadata: Optional patient metadata dictionary
            enable_shap: Whether to generate SHAP explanations (can be slow)
            lime_samples: Number of samples for LIME explanation
            
        Returns:
            ExplanationArtifacts object with paths to all generated files
        """
        print(f"Generating complete explanation for patient {patient_id}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare image
        original_img, preprocessed_img = self._load_and_prepare_image(image_path)
        
        # Get model prediction
        prediction_result = self._get_model_prediction(preprocessed_img)
        
        print("Generating Grad-CAM explanations...")
        # Generate Grad-CAM explanations
        gradcam_heatmap, gradcam_overlay = generate_gradcam_explanation(
            self.model, preprocessed_img, original_img, patient_id, output_dir
        )
        
        print("Generating LIME explanations...")
        # Generate LIME explanations
        lime_positive, lime_negative = generate_lime_explanation(
            self.model, original_img, self.preprocess_fn, patient_id, 
            output_dir, lime_samples
        )
        
        # Generate SHAP explanations (optional, as it can be slow)
        shap_waterfall = None
        if enable_shap:
            print("Generating SHAP explanations...")
            try:
                shap_waterfall = generate_single_shap_explanation(
                    self.model, preprocessed_img, patient_id, output_dir
                )
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
                print("Continuing without SHAP explanations...")
        
        print("Generating probability dashboard...")
        # Generate probability dashboard
        probability_chart = generate_probability_dashboard(
            prediction_result['probabilities'], patient_id, output_dir, 
            self.class_names, metadata
        )
        
        print("Generating HTML report...")
        # Generate comprehensive HTML report
        artifacts_dict = {
            'gradcam_heatmap': gradcam_heatmap,
            'gradcam_overlay': gradcam_overlay,
            'lime_positive': lime_positive,
            'lime_negative': lime_negative,
            'shap_waterfall': shap_waterfall,
            'probability_chart': probability_chart
        }
        
        html_report = generate_html_report(
            patient_id, artifacts_dict, output_dir, metadata, prediction_result
        )
        
        # Create and return artifacts object
        artifacts = ExplanationArtifacts(
            gradcam_heatmap=gradcam_heatmap,
            gradcam_overlay=gradcam_overlay,
            lime_positive=lime_positive,
            lime_negative=lime_negative,
            shap_waterfall=shap_waterfall,
            probability_chart=probability_chart,
            html_report=html_report,
            patient_id=patient_id
        )
        
        print(f"Complete explanation generated for patient {patient_id}")
        print(f"HTML report available at: {html_report}")
        
        return artifacts
    
    def batch_explain(self, dataset: List[Dict], output_dir: str,
                     enable_shap: bool = True, lime_samples: int = 1000) -> pd.DataFrame:
        """
        Process multiple chest X-ray images in batch mode.
        
        Args:
            dataset: List of dictionaries with 'image_path', 'patient_id', and optional 'metadata'
            output_dir: Base output directory for all results
            enable_shap: Whether to generate SHAP explanations
            lime_samples: Number of samples for LIME explanations
            
        Returns:
            DataFrame with patient_id and paths to all generated artifacts
        """
        print(f"Starting batch processing for {len(dataset)} patients...")
        
        results = []
        
        for i, item in enumerate(dataset, 1):
            patient_id = item['patient_id']
            image_path = item['image_path']
            metadata = item.get('metadata', {})
            
            print(f"Processing patient {i}/{len(dataset)}: {patient_id}")
            
            try:
                # Create patient-specific output directory
                patient_output_dir = os.path.join(output_dir, f"patient_{patient_id}")
                
                # Generate explanations
                artifacts = self.generate_complete_explanation(
                    image_path, patient_id, patient_output_dir, metadata,
                    enable_shap, lime_samples
                )
                
                # Add to results
                result_row = {'patient_id': patient_id}
                result_row.update(artifacts.to_dict())
                results.append(result_row)
                
                print(f"✓ Completed patient {patient_id}")
                
            except Exception as e:
                print(f"✗ Failed to process patient {patient_id}: {e}")
                # Add failed entry to results
                results.append({
                    'patient_id': patient_id,
                    'error': str(e)
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save manifest CSV
        manifest_path = os.path.join(output_dir, 'batch_results_manifest.csv')
        results_df.to_csv(manifest_path, index=False)
        
        print(f"Batch processing complete. Manifest saved to: {manifest_path}")
        
        return results_df


# High-level convenience functions for external API

def generate_complete_explanation(model: tf.keras.Model, image_path: str,
                                patient_id: str, class_names: List[str],
                                output_dir: str, preprocess_fn: Callable = None,
                                metadata: Dict = None) -> ExplanationArtifacts:
    """
    High-level function to generate complete explanation with single call.
    
    Args:
        model: TensorFlow/Keras model
        image_path: Path to chest X-ray image
        patient_id: Patient identifier
        class_names: List of class names
        output_dir: Output directory for artifacts
        preprocess_fn: Preprocessing function (if None, uses default)
        metadata: Optional patient metadata
        
    Returns:
        ExplanationArtifacts object
    """
    if preprocess_fn is None:
        # Default preprocessing function
        def default_preprocess(pil_img: Image.Image) -> np.ndarray:
            img = pil_img.convert("RGB").resize((224, 224))
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
        
        preprocess_fn = default_preprocess
    
    pipeline = ExplainabilityPipeline(model, preprocess_fn, class_names)
    
    return pipeline.generate_complete_explanation(
        image_path, patient_id, output_dir, metadata
    )


def batch_explain(model: tf.keras.Model, dataset: List[Dict],
                 output_dir: str, class_names: List[str] = None,
                 preprocess_fn: Callable = None) -> pd.DataFrame:
    """
    High-level function for batch processing.
    
    Args:
        model: TensorFlow/Keras model
        dataset: List of dicts with image_path, patient_id, metadata
        output_dir: Base output directory
        class_names: List of class names
        preprocess_fn: Preprocessing function
        
    Returns:
        DataFrame with results manifest
    """
    if preprocess_fn is None:
        # Default preprocessing function
        def default_preprocess(pil_img: Image.Image) -> np.ndarray:
            img = pil_img.convert("RGB").resize((224, 224))
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
        
        preprocess_fn = default_preprocess
    
    pipeline = ExplainabilityPipeline(model, preprocess_fn, class_names)
    
    return pipeline.batch_explain(dataset, output_dir)