"""
HTML Report Generator for chest X-ray pneumonia detection explanations.

Compiles all explanation artifacts (Grad-CAM, LIME, SHAP, probability charts)
into comprehensive HTML reports for individual patients.
"""

import os
import base64
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class HTMLReportGenerator:
    """
    Generates comprehensive HTML reports combining all explanation artifacts.
    """
    
    def __init__(self):
        """Initialize the HTML report generator."""
        self.template = self._get_html_template()
    
    def _get_html_template(self) -> str:
        """
        Get the HTML template for the report.
        
        Returns:
            HTML template string
        """
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chest X-ray Pneumonia Detection Report - {patient_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #4a5568;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .section-description {{
            background-color: #f7fafc;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
            border-radius: 0 5px 5px 0;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 20px 0;
        }}
        .metadata {{
            background-color: #edf2f7;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .metadata h3 {{
            margin-top: 0;
            color: #2d3748;
        }}
        .metadata-item {{
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #4a5568;
        }}
        .footer {{
            background-color: #2d3748;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .prediction-result {{
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .prediction-result.pneumonia {{
            background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        }}
        .prediction-result h3 {{
            margin: 0 0 10px 0;
            font-size: 1.5em;
        }}
        .confidence-score {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Chest X-ray Analysis Report</h1>
            <p>Patient ID: {patient_id} | Generated: {timestamp}</p>
        </div>
        
        <div class="content">
            {metadata_section}
            
            {prediction_section}
            
            <div class="section">
                <h2>📊 Prediction Probabilities</h2>
                <div class="section-description">
                    <p>This dashboard shows the model's confidence in its predictions and analyzes contributing factors that influenced the decision.</p>
                </div>
                <div class="image-container">
                    {probability_chart}
                </div>
            </div>
            
            <div class="section">
                <h2>🔥 Grad-CAM Attention Analysis</h2>
                <div class="section-description">
                    <p>Grad-CAM highlights the regions of the X-ray that the neural network focuses on when making its prediction. Red areas indicate high attention, while blue areas indicate lower attention.</p>
                </div>
                <div class="grid">
                    <div>
                        <h4>Attention Heatmap</h4>
                        <div class="image-container">
                            {gradcam_heatmap}
                        </div>
                    </div>
                    <div>
                        <h4>Overlay on Original Image</h4>
                        <div class="image-container">
                            {gradcam_overlay}
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 LIME Superpixel Explanations</h2>
                <div class="section-description">
                    <p>LIME segments the image into superpixels and identifies which regions support (green) or contradict (red) the pneumonia diagnosis.</p>
                </div>
                <div class="grid">
                    <div>
                        <h4>Positive Contributing Regions</h4>
                        <div class="image-container">
                            {lime_positive}
                        </div>
                    </div>
                    <div>
                        <h4>Negative Contributing Regions</h4>
                        <div class="image-container">
                            {lime_negative}
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 SHAP Feature Importance</h2>
                <div class="section-description">
                    <p>SHAP (SHapley Additive exPlanations) provides a unified approach to explain individual predictions by quantifying the contribution of each feature.</p>
                </div>
                <div class="image-container">
                    {shap_waterfall}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by Chest X-ray Pneumonia Detection Explainability System</p>
            <p>Report created on {timestamp}</p>
        </div>
    </div>
</body>
</html>
        """
    
    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        Encode an image to base64 for embedding in HTML.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string or None if file doesn't exist
        """
        if not os.path.exists(image_path):
            return None
        
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                img_ext = Path(image_path).suffix.lower()
                
                if img_ext in ['.png']:
                    mime_type = 'image/png'
                elif img_ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                else:
                    mime_type = 'image/png'  # Default
                
                return f"data:{mime_type};base64,{img_base64}"
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def _create_image_tag(self, image_path: str, alt_text: str = "Analysis Image") -> str:
        """
        Create an HTML img tag with base64 encoded image.
        
        Args:
            image_path: Path to the image
            alt_text: Alternative text for the image
            
        Returns:
            HTML img tag or placeholder text
        """
        base64_img = self._encode_image_to_base64(image_path)
        
        if base64_img:
            return f'<img src="{base64_img}" alt="{alt_text}" style="max-width: 100%; height: auto;">'
        else:
            return f'<div style="padding: 50px; background-color: #f0f0f0; text-align: center; border-radius: 8px;">Image not available: {alt_text}</div>'
    
    def _create_metadata_section(self, metadata: Dict) -> str:
        """
        Create the metadata section of the report.
        
        Args:
            metadata: Dictionary containing patient metadata
            
        Returns:
            HTML string for metadata section
        """
        if not metadata:
            return ""
        
        metadata_items = []
        for key, value in metadata.items():
            formatted_key = key.replace('_', ' ').title()
            metadata_items.append(f'''
                <div class="metadata-item">
                    <span class="metadata-label">{formatted_key}:</span>
                    <span>{value}</span>
                </div>
            ''')
        
        return f'''
        <div class="metadata">
            <h3>Patient Information</h3>
            {''.join(metadata_items)}
        </div>
        '''
    
    def _create_prediction_section(self, prediction_result: Dict) -> str:
        """
        Create the prediction result section.
        
        Args:
            prediction_result: Dictionary containing prediction info
            
        Returns:
            HTML string for prediction section
        """
        diagnosis = prediction_result.get('diagnosis', 'Unknown')
        confidence = prediction_result.get('confidence', 0)
        probability = prediction_result.get('raw_probability', 0.5)
        
        css_class = 'pneumonia' if diagnosis.lower() == 'pneumonia' else ''
        
        return f'''
        <div class="prediction-result {css_class}">
            <h3>Primary Diagnosis</h3>
            <div class="confidence-score">{diagnosis}</div>
            <p>Confidence: {confidence:.1f}% | Raw Probability: {probability:.3f}</p>
        </div>
        '''
    
    def generate_report(self, patient_id: str, artifacts: Dict[str, str],
                       output_path: str, metadata: Dict = None,
                       prediction_result: Dict = None) -> str:
        """
        Generate a complete HTML report.
        
        Args:
            patient_id: Patient identifier
            artifacts: Dictionary mapping artifact types to file paths
            output_path: Path to save the HTML report
            metadata: Optional patient metadata
            prediction_result: Optional prediction result dictionary
            
        Returns:
            Path to generated HTML report
        """
        # Create image tags for all artifacts
        gradcam_heatmap = self._create_image_tag(
            artifacts.get('gradcam_heatmap', ''), 'Grad-CAM Heatmap'
        )
        gradcam_overlay = self._create_image_tag(
            artifacts.get('gradcam_overlay', ''), 'Grad-CAM Overlay'
        )
        lime_positive = self._create_image_tag(
            artifacts.get('lime_positive', ''), 'LIME Positive Regions'
        )
        lime_negative = self._create_image_tag(
            artifacts.get('lime_negative', ''), 'LIME Negative Regions'
        )
        shap_waterfall = self._create_image_tag(
            artifacts.get('shap_waterfall', ''), 'SHAP Waterfall Plot'
        )
        probability_chart = self._create_image_tag(
            artifacts.get('probability_chart', ''), 'Probability Dashboard'
        )
        
        # Create sections
        metadata_section = self._create_metadata_section(metadata or {})
        prediction_section = self._create_prediction_section(prediction_result or {})
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Fill template
        html_content = self.template.format(
            patient_id=patient_id,
            timestamp=timestamp,
            metadata_section=metadata_section,
            prediction_section=prediction_section,
            gradcam_heatmap=gradcam_heatmap,
            gradcam_overlay=gradcam_overlay,
            lime_positive=lime_positive,
            lime_negative=lime_negative,
            shap_waterfall=shap_waterfall,
            probability_chart=probability_chart
        )
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path


def generate_html_report(patient_id: str, artifacts: Dict[str, str], 
                        output_dir: str, metadata: Dict = None,
                        prediction_result: Dict = None) -> str:
    """
    High-level function to generate HTML report.
    
    Args:
        patient_id: Patient identifier
        artifacts: Dictionary of artifact paths
        output_dir: Output directory for the report
        metadata: Optional patient metadata
        prediction_result: Optional prediction result
        
    Returns:
        Path to generated HTML report
    """
    generator = HTMLReportGenerator()
    
    report_path = os.path.join(output_dir, f"Report_{patient_id}.html")
    
    return generator.generate_report(
        patient_id, artifacts, report_path, metadata, prediction_result
    )