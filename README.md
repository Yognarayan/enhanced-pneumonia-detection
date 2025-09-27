# 🫁 Chest X-ray Pneumonia Detection with AI Explainability

A comprehensive TensorFlow-based chest X-ray pneumonia detection system enhanced with state-of-the-art explainability features including SHAP, LIME, Grad-CAM, and automated reporting.

## ✨ Key Features

- **🎯 High-Accuracy Classification**: Pre-trained CNN model for pneumonia detection
- **🔥 Grad-CAM Attention Maps**: Visual heatmaps showing model focus areas
- **🔍 LIME Superpixel Analysis**: Local explanations with positive/negative regions
- **📊 SHAP Feature Importance**: Global and local feature attribution analysis
- **📈 Probability Dashboard**: Clean confidence metrics and contributing factors
- **📄 Automated HTML Reports**: Professional reports combining all explanations
- **⚡ Batch Processing**: Process multiple patients automatically
- **🖥️ Interactive Web Interface**: Gradio-based GUI for easy use
- **🛠️ Command Line Interface**: CLI for batch processing and automation

## 🚀 Quick Start

### Installation

1. **Clone or download the project**
```bash
git clone <repository-url>
cd Pneumonia-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

#### 🖥️ Web Interface (Gradio)
```bash
python pheumonia/pneumonia.py
```
This launches an interactive web interface where you can:
- Upload chest X-ray images
- Get instant pneumonia predictions  
- Generate comprehensive AI explanations
- View detailed analysis reports

#### 🛠️ Command Line Interface

**Process a single image:**
```bash
python -m explain.run --image chest_xray.jpg --patient-id P001 --output-dir ./results
```

**Process with metadata:**
```bash
python -m explain.run --image chest_xray.jpg --patient-id P001 --output-dir ./results \
  --metadata age=45 gender=M symptoms="cough,fever"
```

**Batch process from CSV:**
```bash
python -m explain.run --batch-csv patients.csv --output-dir ./batch_results
```

**Create sample CSV template:**
```bash
python -m explain.run --create-sample-csv sample_patients.csv
```

### 📊 Programmatic API

```python
from explain.pipeline import generate_complete_explanation
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('path/to/model.h5')

# Generate complete explanation
artifacts = generate_complete_explanation(
    model=model,
    image_path='chest_xray.jpg',
    patient_id='P001', 
    class_names=['Normal', 'Pneumonia'],
    output_dir='./results'
)

print(f"HTML Report: {artifacts.html_report}")
print(f"Grad-CAM Heatmap: {artifacts.gradcam_heatmap}")
```

## 📁 Project Structure

```
Pneumonia-detection/
├── explain/                     # Explainability package
│   ├── __init__.py             # Package initialization
│   ├── gradcam.py              # Grad-CAM implementation
│   ├── lime_explain.py         # LIME explanations
│   ├── shap_explain.py         # SHAP feature importance
│   ├── dashboard.py            # Probability visualizations
│   ├── report.py               # HTML report generation
│   ├── pipeline.py             # Main pipeline orchestration
│   └── run.py                  # CLI interface
├── pheumonia/
│   └── pneumonia.py            # Enhanced Gradio interface
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔬 Explanation Methods

### 1. 🔥 Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Purpose**: Shows which regions of the X-ray the model focuses on
- **Output**: Heatmap overlay on original image
- **Files Generated**: 
  - `GradCAM_Heatmap_{patient_id}.png`
  - `GradCAM_Overlay_{patient_id}.png`

### 2. 🔍 LIME (Local Interpretable Model-agnostic Explanations) 
- **Purpose**: Explains individual predictions using superpixel segmentation
- **Output**: Positive and negative contributing regions
- **Files Generated**:
  - `LIME_Positive_{patient_id}.png`
  - `LIME_Negative_{patient_id}.png`

### 3. 📊 SHAP (SHapley Additive exPlanations)
- **Purpose**: Quantifies feature contributions using game theory
- **Output**: Waterfall plots showing feature importance
- **Files Generated**:
  - `SHAP_Waterfall_{patient_id}.png`
  - `SHAP_Summary_{batch_id}.png` (for batches)

### 4. 📈 Probability Dashboard
- **Purpose**: Clean visualization of prediction confidence and contributing factors
- **Output**: Professional probability charts with metadata analysis
- **Files Generated**: `Probability_Dashboard_{patient_id}.png`

### 5. 📄 HTML Report
- **Purpose**: Comprehensive report combining all explanations
- **Output**: Self-contained HTML file with embedded images
- **Files Generated**: `Report_{patient_id}.html`

## 📋 CSV Format for Batch Processing

Create a CSV file with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `patient_id` | ✅ Yes | Unique patient identifier |
| `image_path` | ✅ Yes | Path to chest X-ray image |
| `age` | ❌ Optional | Patient age |
| `gender` | ❌ Optional | Patient gender |
| `symptoms` | ❌ Optional | Clinical symptoms |
| `*` | ❌ Optional | Any additional metadata |

**Example CSV:**
```csv
patient_id,image_path,age,gender,symptoms
P001,/data/xrays/patient_001.jpg,45,M,"cough, fever"
P002,/data/xrays/patient_002.jpg,62,F,"shortness of breath"
P003,/data/xrays/patient_003.jpg,38,M,"chest pain"
```

## 🔧 Configuration Options

### CLI Options
```bash
--image PATH              # Single image path
--batch-csv PATH          # CSV file for batch processing  
--patient-id ID           # Patient identifier
--output-dir DIR          # Output directory (default: ./explanation_results)
--metadata KEY=VALUE      # Additional metadata (multiple allowed)
--disable-shap            # Skip SHAP explanations (faster)
--lime-samples N          # Number of LIME samples (default: 1000)
```

### Programmatic Options
```python
generate_complete_explanation(
    model=model,              # TensorFlow/Keras model
    image_path=path,          # Path to chest X-ray
    patient_id=id,            # Patient identifier  
    class_names=classes,      # ['Normal', 'Pneumonia']
    output_dir=dir,           # Output directory
    preprocess_fn=func,       # Preprocessing function
    metadata=dict             # Patient metadata
)
```

## 📊 Output Examples

After processing, you'll find:

```
results/
├── patient_P001/
│   ├── GradCAM_Heatmap_P001.png      # CNN attention heatmap
│   ├── GradCAM_Overlay_P001.png      # Heatmap overlaid on X-ray
│   ├── LIME_Positive_P001.png        # Regions supporting pneumonia
│   ├── LIME_Negative_P001.png        # Regions contradicting pneumonia
│   ├── SHAP_Waterfall_P001.png       # Feature importance breakdown
│   ├── Probability_Dashboard_P001.png # Confidence visualization
│   └── Report_P001.html               # Complete analysis report
└── batch_results_manifest.csv        # Batch processing summary
```

## 🧠 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Training Data**: Chest X-ray images from public datasets
- **Classes**: Normal vs Pneumonia (binary classification)
- **Input Size**: 224x224x3 RGB images
- **Performance**: ~94% accuracy on validation set

## ⚙️ Requirements

### Core Dependencies
- `tensorflow>=2.10.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Plotting and visualization
- `scikit-image>=0.19.0` - Image processing

### Explainability Libraries  
- `shap>=0.41.0` - SHAP explanations
- `lime>=0.2.0` - LIME explanations
- `opencv-python>=4.6.0` - Computer vision utilities

### Additional Libraries
- `gradio>=3.40.0` - Web interface
- `pandas>=1.4.0` - Data manipulation
- `Pillow>=9.0.0` - Image processing
- `huggingface_hub>=0.15.0` - Model loading

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

**2. SHAP Explanations Slow/Failing**
```bash
python -m explain.run --image chest_xray.jpg --patient-id P001 --disable-shap
```

**3. Memory Issues with Large Batches**
- Process smaller batches
- Reduce `--lime-samples` parameter
- Use `--disable-shap` for faster processing

**4. GPU Memory Issues**
```python
# Add to your script
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original pneumonia detection model from Hugging Face
- SHAP library for explainable AI
- LIME library for local explanations
- Grad-CAM implementation based on original paper
- TensorFlow and Keras communities

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed description
4. Include error messages and system information

---

**⚡ Quick Commands Reference:**
```bash
# Install dependencies
pip install -r requirements.txt

# Launch web interface  
python pheumonia/pneumonia.py

# Process single image
python -m explain.run --image chest_xray.jpg --patient-id P001 --output-dir ./results

# Batch processing
python -m explain.run --batch-csv patients.csv --output-dir ./batch_results

# Create sample CSV
python -m explain.run --create-sample-csv sample.csv
```